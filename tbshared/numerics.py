"""Numerics guard: the NUMERICS fields of every model-serving spec of a pipeline, as a stable static record.

  python -m tbshared.numerics extract PIPELINE ROOT [--ref REF]   # print the extraction (JSON)
  python -m tbshared.numerics record PIPELINE ROOT [--ref REF]    # update that pipeline's section of numerics_record.json
  python -m tbshared.numerics check [PIPELINE] [ROOT]              # what test_numerics.py checks (exit 1 on a change)
  python -m tbshared.numerics diff [--campaigns]                   # cross-pipeline table for models in >1 pipeline

PIPELINE is one of PIPELINES; ROOT is a checkout of that pipeline's branch (with --ref REF, the files are read from
git, `git -C ROOT show REF:path`, so a branch head can be recorded without a worktree). Stdlib only (tomllib: Python
3.11+; on 3.10 TOML specs are skipped with a message, so the TOML pipelines cannot be checked there).

What counts as numerics (README "Numerics guard"): weights (checkpoint, revision, declared and effective weight
quantization), KV and Mamba/state cache dtypes, attention and kernel flags, numerics-relevant environment, max model
len, the output cap and whether it is hard (`max_new_tokens`) or a default (`max_tokens`) or client-side, chat template
(path, sha256 when the file is in the repo) and its default kwargs (thinking flags), reasoning/tool parsers, sampling
defaults (server --override-generation-config and client request overrides), extra stop tokens, seed, image limits, the
vLLM runtime (venv/SIF, declared version, CUDA home, entrypoint), and speculative decoding (`adjacent.*`: numerics-
adjacent, the distribution is meant to stay the same). Throughput fields (memory utilization, max seqs, batched tokens,
prefix caching, TP/DP/EP layout, the CPU KV tier, gates, resources) are left out. A vLLM flag this module does not know
is recorded as `args.<flag>` (conservative: an unknown flag counts as numerics until it is classified here).

Sources, per pipeline:
- aa-serving, task-synthesis: synth/teachers/*.toml under task_synthesis/serve/serve_teacher.py's PROFILES / PROTOCOL
  (read with ast, never imported; one spec per replica_tunables arm where serve_teacher reads them), and every
  synth/campaigns/*.toml as synth/serve.py serves it (teacher_overrides, [protocol], tuning candidates). Each is
  rendered with the branch's own synth/vllm_launch.replica_command and replica_env (the cu13-wheel CUDA_HOME stubbed).
  The weight-quantization label comes from the branch's synth/numerics.weight_quantization when it exists.
- eval: eval_q35/submit.py's `init` (its config.json construction, executed with stubbed I/O and default arguments
  per family x serving variant x GPU set x harness; the real Pi finish notice from tb_agents/pi_context_limit.py),
  then eval_q35/serve_job.sh's argument building and exports (a small interpreter; whatever it cannot follow that
  touches the vLLM arguments becomes a warning, and warnings are compared like fields), with the quantization label
  from eval_q35/serve_numerics.weight_quantization. Per-run inputs (checkpoint, --chat-template, --vllm-extra-args,
  --quantization) are placeholders.
- sft is not covered yet (its BPB / serve_check scripts have no spec files; port_matrix_20261114/serving/report.md).

These are the specs as written. The pipelines' runtime records (aa-serving synth/numerics.py -> numerics.<job>.json,
eval eval_q35/serve_numerics.py -> RUN/serve/numerics.<tag>.json) are the effective numerics of each job after every
override; this record is what a spec change is checked against before any job runs.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import concurrent.futures
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ImportError:              # Python 3.10: no TOML specs (README)
    tomllib = None

HERE = Path(__file__).resolve().parent
RECORD = HERE / 'numerics_record.json'
PIPELINES = ('aa-serving', 'task-synthesis', 'eval')
SCHEMA = 1
CHANGE_HINT = ("a numerics change needs a new eval baseline (user's call); if intended, update the record in the same "
               "change (on the trunk `main`: a PR into main; on a vendored branch: on branch `shared`, then bump), "
               "or right before it merges (python -m tbshared.numerics record <pipeline> <repo> --ref <the change's "
               "commit>)")

# Differences the trunk `main` is known to have against a record section, each with why (#480). `check` and
# test_numerics report them apart and fail on anything else, so a new numerics change on main still fails. Remove an
# entry when its cause is gone and the section is re-recorded from main (accepting a field change is the user's
# call). Rule kinds: ('re', pattern, reason) on a diff line; ('campaigns-unrenderable', None, reason), ('union',
# section, reason) and ('pinned', spec, reason): see known_trunk_gaps.
# Empty since the serve_teacher sections were re-recorded from main 77898608 (#480, after uni-agents' tb_agents and
# #486/#490): main matches every section.
TRUNK_KNOWN_GAPS: dict = {'aa-serving': [], 'task-synthesis': []}
# ('pinned', spec) entries: the sha256 of the spec's repository-derived fields (host.* left out, so a node without
# /data does not fail), python -m tbshared.numerics digest PIPELINE SPEC [ROOT] prints it.
TRUNK_PINNED: dict = {}
_CAMPAIGN_SKIP = re.compile(r"^skipped \(could not extract\): synth/campaigns/([^ ]+)\.toml .*cannot import name "
                            r"'finish_notice_record'")


def spec_digest(fields: dict) -> str:
    """sha256 of a spec's repository-derived fields (host.* left out)."""
    return hashlib.sha256(json.dumps(_jsonable({k: v for k, v in fields.items() if not k.startswith(HOST)}),
                                     sort_keys=True).encode()).hexdigest()


def known_trunk_gaps(pipeline: str, diff: list[str], record_path=None, specs: dict | None = None,
                     rules=None, pinned=None) -> tuple[list[str], list[tuple[str, str]]]:
    """Split a trunk check's lines into (unexplained, [(line, reason)]) by TRUNK_KNOWN_GAPS[pipeline]:
    - campaigns-unrenderable: a campaign file skipped because tb_agents lacks finish_notice_record, and the removed
      specs of exactly those campaigns (campaign/<name> and its candidates);
    - union: an added spec that the named section records with the same fields (`specs`: the extraction; without it
      no added spec is explained);
    - pinned: an added spec whose repository-derived fields hash to TRUNK_PINNED[spec] (`specs` needed too);
    - re: a line matching the pattern."""
    rules = TRUNK_KNOWN_GAPS.get(pipeline, []) if rules is None else rules
    pinned = TRUNK_PINNED if pinned is None else pinned
    skipped = {m.group(1) for line in diff if (m := _CAMPAIGN_SKIP.match(line))}
    other = {}
    for kind, arg, _ in rules:
        if kind == 'union':
            other.update(load_record(record_path or RECORD)['pipelines'].get(arg, {}).get('specs', {}))
    left, known = [], []
    for line in diff:
        reason = None
        for kind, arg, why in rules:
            if kind == 'campaigns-unrenderable':
                m = re.match(r'^removed spec campaign/([^/ ]+)', line)
                if _CAMPAIGN_SKIP.match(line) or (m and m.group(1) in skipped):
                    reason = why
            elif kind == 'union':
                m = re.match(r'^added spec ([^:]+):', line)
                if m and m.group(1) in other and specs is not None and m.group(1) in specs and \
                        not compare({m.group(1): other[m.group(1)]}, {m.group(1): specs[m.group(1)]}, []):
                    reason = why
            elif kind == 'pinned':
                if line.startswith(f'added spec {arg}: ') and specs is not None and arg in specs and \
                        spec_digest(specs[arg]) == pinned.get(arg):
                    reason = why
            elif re.search(arg, line):
                reason = why
            if reason:
                break
        (known.append((line, reason)) if reason else left.append(line))
    return left, known

# ---------------------------------------------------------------------------------------------------------------------
# Classification of vLLM flags and environment variables
# ---------------------------------------------------------------------------------------------------------------------
# Throughput, layout, logging and plumbing flags: not recorded. Prefix entries end with '*'. `--block-size` is NOT
# here on purpose: the KV page size changes kernel choice (paged-attention tiles, MLA / hybrid Mamba block alignment)
# and where prefix-cache hits start, so it stays recorded (as args.--block-size).
THROUGHPUT_FLAGS = {
    '--host', '--port', '--served-model-name', '--model', '--gpu-memory-utilization', '--max-num-seqs',
    '--max-num-batched-tokens', '--tensor-parallel-size', '--data-parallel-*', '--dp-supervisor-probe-timeout-s',
    '--enable-expert-parallel', '--enable-ep-weight-filter', '--api-server-count', '--enable-prefix-caching',
    '--no-enable-prefix-caching', '--enable-chunked-prefill', '--prefix-match-unit',
    '--enable-mamba-shared-prefix-checkpoint', '--kv-offloading-backend', '--kv-offloading-size', '--kv-transfer-config',
    '--enable-prompt-tokens-details', '--no-enable-log-requests', '--enable-log-requests', '--disable-log-requests',
    '--disable-log-stats', '--uvicorn-log-level', '--disable-uvicorn-access-log', '--max-log-len',
    '--max-cudagraph-capture-size', '--async-scheduling',
    '--renderer-num-workers',   # the API server's tokenize / chat-template thread pool (#451 ws4)
}
# Flags skipped only at values known not to change outputs (anything else is recorded as args.<flag>): weight
# loading through the normal safetensors path (`--load-format dummy` loads random weights), the multithreaded loader,
# and DeepSeek's Engram table placement.
SAFE_VALUES = {'--load-format': {'auto', 'safetensors', 'fastsafetensors'}}
SAFE_JSON_KEYS = {'--model-loader-extra-config': {'enable_multithread_load', 'num_threads'},
                  '--engram-config': {'cpu_offload', 'embedding_across_dp', 'dp_shared_memory'}}
# --compilation-config keys that only choose which batch sizes get a CUDA graph (padding up to the next captured size;
# eager below and above): left out of compilation_config.*.
COMPILATION_THROUGHPUT_KEYS = {'cudagraph_capture_sizes', 'decoder_replay_cudagraph_capture_sizes',
                               'max_cudagraph_capture_size', 'max_capture_size'}
# Numerics flags with a named field (and the default recorded when the flag is absent: what vLLM then does).
NAMED_FLAGS = {
    '--quantization': ('quantization', 'none'),
    '--kv-cache-dtype': ('kv_cache_dtype', 'auto'),
    '--mamba-ssm-cache-dtype': ('mamba_ssm_cache_dtype', 'auto'),
    '--mamba-cache-dtype': ('mamba_cache_dtype', 'auto'),
    '--attention-backend': ('attention_backend', 'vllm default'),
    '--max-model-len': ('max_model_len', None),
    '--generation-config': ('generation_config', 'auto'),
    '--reasoning-parser': ('reasoning_parser', 'none'),
    '--tool-call-parser': ('tool_parser', 'none'),
    '--reasoning-parser-plugin': ('reasoning_parser_plugin', None),
    '--tool-parser-plugin': ('tool_parser_plugin', None),
    '--enable-auto-tool-choice': ('auto_tool_choice', False),
    '--seed': ('seed', 'vllm default'),
    '--tokenizer': ('weights.tokenizer', None),
    '--mm-processor-kwargs': ('mm_processor_kwargs', None),
    '--trust-remote-code': ('trust_remote_code', False),
}
JSON_FLAGS = {'--attention-config': 'attention_config', '--default-chat-template-kwargs': 'chat_template_kwargs',
              '--speculative-config': 'adjacent.speculative', '--hf-overrides': 'hf_overrides',
              '--compilation-config': 'compilation_config'}
# Environment the launchers set that does not change outputs (offline switches, threads, timeouts, caches, paths,
# multiprocessing method, the CPU KV tier's connector switch, NVLS). CUDA_HOME is recorded as vllm.cuda_home; the
# cpu-supersede switch as adjacent.cpu_supersede (it changes eviction order only; its PYTHONPATH entry stays numerics).
ENV_SKIP = {
    'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_XET', 'TRANSFORMERS_OFFLINE', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'OMP_NUM_THREADS', 'NCCL_NVLS_ENABLE', 'PYTHONDONTWRITEBYTECODE', 'PYTHONNOUSERSITE', 'TOKENIZERS_PARALLELISM',
    'FLASHINFER_NO_DOWNLOAD',   # a missing cubin fails instead of downloading (the artifacts are the same)
    'VLLM_ENGINE_READY_TIMEOUT_S', 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS', 'VLLM_RPC_TIMEOUT',
    'VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES', 'VLLM_USE_SIMPLE_KV_OFFLOAD', 'VLLM_WORKER_MULTIPROC_METHOD',
    'HUMMING_DISABLE_PARALLEL_BUILD', 'HOME', 'USER', 'TMPDIR', 'PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH',
    'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'FLASHINFER_WORKSPACE_BASE',
    'HF_HOME', 'HUMMING_CACHE_DIR', 'HUMMING_TMP_DIR', 'CUDA_VISIBLE_DEVICES', 'no_proxy', 'NO_PROXY', 'CUDA_HOME',
    'NVCC_APPEND_FLAGS', 'SYNTH_CPU_SUPERSEDE', 'CUDA_DEVICE_ORDER',
}
# Environment values that are node paths whose content, not location, matters: recorded as a cheap content identity.
ENV_DIRS = {'GRUG_FI_AOT_DIR'}


class ExtractError(RuntimeError):
    """A spec or source file the extractor cannot read: fix the extractor on branch `shared` (or the source)."""


def _throughput(flag: str) -> bool:
    return flag in THROUGHPUT_FLAGS or any(flag.startswith(p[:-1]) for p in THROUGHPUT_FLAGS if p.endswith('*'))


def flatten(prefix: str, value, out: dict):
    """Dicts become dotted keys (prefix.key...); everything else is one value."""
    if isinstance(value, dict) and value:
        for k in sorted(value):
            flatten(f'{prefix}.{k}', value[k], out)
    else:
        out[prefix] = value
    return out


def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def parse_argv(argv) -> dict:
    """{flag: value} of a vLLM argv (last occurrence wins, as in vLLM; `--a=b` and `--a.b=c` forms; a flag followed
    by another flag or the end is a switch, True). Positionals (entrypoint, model) are dropped."""
    out = {}
    i = 0
    while i < len(argv):
        a = str(argv[i])
        if not a.startswith('--'):
            i += 1
            continue
        if '=' in a:
            k, v = a.split('=', 1)
            out[k] = v
            i += 1
        elif i + 1 < len(argv) and not str(argv[i + 1]).startswith('--'):
            out[a] = str(argv[i + 1])
            i += 2
        else:
            out[a] = True
            i += 1
    return out


def argv_fields(argv) -> tuple[dict, list[str]]:
    """The NUMERICS fields of a vLLM argv (see the module docstring) and warnings."""
    flags, fields, warn = parse_argv(argv), {}, []
    for flag, (name, default) in NAMED_FLAGS.items():
        if flag in flags:
            fields[name] = flags[flag]
        elif default is not None:
            fields[name] = default
    lm_only = flags.get('--language-model-only') is True
    if lm_only:
        fields['image_limits'] = 'text only (--language-model-only)'
    elif '--limit-mm-per-prompt' in flags:
        v = _json(flags['--limit-mm-per-prompt'])
        fields['image_limits'] = v if v is not None else flags['--limit-mm-per-prompt']
    else:
        fields['image_limits'] = 'vllm default (model capabilities)'
    ogc = flags.get('--override-generation-config')
    over = _json(ogc) if ogc else {}
    if ogc and over is None:
        warn.append(f'--override-generation-config is not JSON: {ogc!r}')
        fields['args.--override-generation-config'] = ogc
        over = {}
    over = dict(over or {})
    if 'max_new_tokens' in over:
        fields['output_cap.tokens'] = over.pop('max_new_tokens')
        fields['output_cap.kind'] = 'hard (max_new_tokens: a request cannot raise it)'
        over.pop('max_tokens', None)
    elif 'max_tokens' in over:
        fields['output_cap.tokens'] = over.pop('max_tokens')
        fields['output_cap.kind'] = 'default (max_tokens: a request may raise it)'
    else:
        fields['output_cap.kind'] = 'none on the server'
    fields['stop_token_ids'] = over.pop('stop_token_ids', None)
    flatten('sampling', over, fields) if over else fields.__setitem__('sampling', {})
    tpl = flags.get('--chat-template')
    fields['chat_template.path'] = tpl if tpl else 'checkpoint default (no --chat-template)'
    for flag, prefix in JSON_FLAGS.items():
        if flag in flags:
            v = _json(flags[flag])
            if flag == '--compilation-config' and isinstance(v, dict):
                v = {k: x for k, x in v.items() if k not in COMPILATION_THROUGHPUT_KEYS}
            flatten(prefix, v, fields) if isinstance(v, dict) else fields.__setitem__(prefix, flags[flag])
    if '--speculative-config' not in flags:
        fields['adjacent.speculative'] = 'off'
    for flag, v in flags.items():
        if (flag in NAMED_FLAGS or flag in JSON_FLAGS or _throughput(flag) or flag in (
                '--language-model-only', '--limit-mm-per-prompt', '--override-generation-config', '--chat-template')):
            continue
        if flag in SAFE_VALUES and v in SAFE_VALUES[flag]:
            continue
        if flag in SAFE_JSON_KEYS and isinstance(_json(v), dict) and set(_json(v)) <= SAFE_JSON_KEYS[flag]:
            continue
        base, _, sub = flag.partition('.')
        if sub and base in JSON_FLAGS:                       # --attention-config.indexer_kv_dtype=fp8
            fields[f'{JSON_FLAGS[base]}.{sub}'] = v
            continue
        fields[f'args.{flag}'] = v
    return fields, warn


HOST = 'host.'          # fields read from the node (/data, venvs), not the repository: see README "Numerics guard"
UNREADABLE = 'unreadable'
SAMPLE = 64 << 10       # bytes per sample of a large file: its head, middle and tail
SMALL = 4 << 20         # files up to this size are hashed whole


def file_identity(path) -> str:
    """sha256 of a small file; of a large one, sha256 of (size, head, middle, tail samples): reads at most 192 KiB,
    and still changes when a shard's weights do (a new SFT step, a re-quantized export, a same-size kernel rebuild).
    Cached per (path, size, mtime); UNREADABLE... when the file cannot be read."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError as e:
        return f'{UNREADABLE}: {type(e).__name__}'
    key = ('identity', str(p), st.st_size, st.st_mtime_ns)
    if key not in _FILE_CACHE:
        h = hashlib.sha256()
        try:
            with open(p, 'rb') as f:
                if st.st_size <= SMALL:
                    h.update(f.read())
                else:
                    h.update(str(st.st_size).encode())
                    for off in (0, st.st_size // 2, st.st_size - SAMPLE):
                        f.seek(off)
                        h.update(f.read(SAMPLE))
        except OSError as e:
            return f'{UNREADABLE}: {type(e).__name__}'
        _FILE_CACHE[key] = h.hexdigest()
    return _FILE_CACHE[key]


def dir_identity(path: str) -> str:
    """A node directory's content identity: sha256 of its (relative path, size, file_identity) listing; never its
    location or mtimes (a copy keeps it, a rebuild of the same size does not)."""
    p = Path(path)
    try:
        files = sorted(f for f in p.rglob('*') if f.is_file())
        rows = [(str(f.relative_to(p)), f.stat().st_size, file_identity(f)) for f in files]
    except OSError as e:
        return f'{UNREADABLE}: {type(e).__name__}'
    if not rows:
        return f'{UNREADABLE}: missing or empty'
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def env_fields(env: dict, skip=()) -> dict:
    """env.* fields (numerics environment only), adjacent.cpu_supersede, and for node directories (ENV_DIRS) a
    host.env.<name> content identity in place of the path."""
    out = {}
    for k, v in sorted(env.items()):
        if k in ENV_SKIP or k in skip:
            continue
        if k in ENV_DIRS:
            out[f'env.{k}'] = 'set'
            out[f'{HOST}env.{k}'] = dir_identity(str(v))
        else:
            out[f'env.{k}'] = str(v)
    out['adjacent.cpu_supersede'] = 'on' if str(env.get('SYNTH_CPU_SUPERSEDE', '')) not in ('', '0') else 'off'
    return out


_FILE_CACHE = {}


# ---------------------------------------------------------------------------------------------------------------------
# Reading a pipeline's tree: a directory, or a git ref (git show / ls-tree, nothing extracted)
# ---------------------------------------------------------------------------------------------------------------------
class Tree:
    """Files of a pipeline checkout: FsTree (a directory) or GitTree (a ref of a repository)."""

    def read_bytes(self, rel: str) -> bytes | None: ...

    def list(self, rel_dir: str) -> list[str]: ...

    def list_recursive(self, rel_dir: str) -> list[str]: ...

    def read_text(self, rel: str) -> str | None:
        b = self.read_bytes(rel)
        return b.decode() if b is not None else None

    def sha256(self, rel: str) -> str | None:
        b = self.read_bytes(rel)
        return hashlib.sha256(b).hexdigest() if b is not None else None

    def module(self, rel: str, name: str):
        """A stdlib-only module of this tree, loaded on its own (never on sys.path, nothing else imported from the
        tree); None when the file is missing."""
        text = self.read_text(rel)
        if text is None:
            return None
        with tempfile.TemporaryDirectory(prefix='tbshared-numerics-') as d:
            p = Path(d) / Path(rel).name
            p.write_text(text)
            spec = importlib.util.spec_from_file_location(f'_tbshared_numerics_{name}', p)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as e:  # noqa: BLE001
                raise ExtractError(f'{rel}: cannot load it as a standalone module ({e!r})') from e
        return mod


class FsTree(Tree):
    """A checkout. In a git worktree only tracked files count (`git ls-files`: an untracked file is not part of the
    pipeline, so it cannot change a hash or add a spec); their working-copy content is what is read."""

    def __init__(self, root):
        self.root = Path(root)
        self.label = str(self.root)
        r = subprocess.run(['git', '-C', str(self.root), 'ls-files', '-z'], capture_output=True, timeout=120,
                           stdin=subprocess.DEVNULL)
        self.tracked = set(r.stdout.decode().split('\0')) - {''} if r.returncode == 0 else None

    def _ok(self, rel):
        return self.tracked is None or rel in self.tracked

    def read_bytes(self, rel):
        p = self.root / rel
        return p.read_bytes() if p.is_file() and self._ok(rel) else None

    def list(self, rel_dir):
        p = self.root / rel_dir
        return sorted(r for r in (f'{rel_dir.rstrip("/")}/{x.name}' for x in p.iterdir() if x.is_file())
                      if self._ok(r)) if p.is_dir() else []

    def list_recursive(self, rel_dir):
        p = self.root / rel_dir
        if not rel_dir or not p.is_dir():
            return []
        return sorted(r for r in (str(x.relative_to(self.root)) for x in p.rglob('*') if x.is_file()
                                  and not ({'__pycache__', '.pytest_cache'} & set(x.parts)) and x.suffix != '.pyc')
                      if self._ok(r))


class GitTree(Tree):
    def __init__(self, repo, ref):
        self.repo, self.ref = str(repo), ref
        self.commit = self._git('rev-parse', '--verify', f'{ref}^{{commit}}').decode().strip()
        self.label = f'{ref} ({self.commit[:10]})'

    def _git(self, *args, ok=False):
        r = subprocess.run(['git', '-C', self.repo, *args], capture_output=True, timeout=120,
                           stdin=subprocess.DEVNULL)
        if r.returncode and not ok:
            raise ExtractError(f'git {" ".join(args)}: {r.stderr.decode().strip()[:300]}')
        return r.stdout if not r.returncode else None

    def read_bytes(self, rel):
        return self._git('cat-file', 'blob', f'{self.commit}:{rel}', ok=True)   # None for a tree or a missing path

    def _blobs(self, rel_dir, *flags):
        out = self._git('ls-tree', *flags, self.commit, rel_dir.rstrip('/') + '/', ok=True)
        rows = [line.split('\t', 1) for line in out.decode().splitlines()] if out else []
        return sorted(path for meta, path in rows if meta.split()[1] == 'blob')

    def list(self, rel_dir):
        return self._blobs(rel_dir)

    def list_recursive(self, rel_dir):
        return self._blobs(rel_dir, '-r') if rel_dir else []


def _tree_digest(tree: Tree, files) -> str:
    return hashlib.sha256(json.dumps([(f, tree.sha256(f)) for f in files]).encode()).hexdigest()


def add_repo_hashes(fields: dict, tree: Tree, prefix: str):
    """For every field whose value (or a list element, or a PATH-like `:` part) is a file or directory of the
    repository written as PREFIX/path (templates, parser plugins, model entry points, plugin dirs): `<field>.sha256`
    of the file, or of the directory's tracked (path, sha256) listing."""
    for key in list(fields):
        vals = fields[key] if isinstance(fields[key], list) else [fields[key]]
        parts = [p for v in vals if isinstance(v, str) for p in v.split(':') if p.startswith(prefix)]
        digests = []
        for p in parts:
            rel = p[len(prefix):].rstrip('/')
            sha = tree.sha256(rel) if rel else None
            if sha is None:
                files = tree.list_recursive(rel)
                sha = _tree_digest(tree, files) if files else 'missing from the repository'
            digests.append(sha)
        if digests:
            fields[f'{key}.sha256'] = digests[0] if len(digests) == 1 else digests


def entry_point_modules(tree: Tree, roots) -> list[str]:
    """Repository modules a plugin dir on PYTHONPATH registers through a `*.dist-info/entry_points.txt` (vLLM loads
    every `vllm.general_plugins` entry at start-up, with no import in the spec): `name = module:attr` lines."""
    out = []
    for root in roots:
        for f in tree.list_recursive(root):
            if f.endswith('.dist-info/entry_points.txt'):
                for line in (tree.read_text(f) or '').splitlines():
                    m = re.match(r'\s*[^#\[=]+=\s*([A-Za-z_][\w.]*)', line)
                    if m:
                        mod = m.group(1).replace('.', '/')
                        out += [c for c in (f'{root}/{mod}.py', f'{root}/{mod}/__init__.py')
                                if tree.read_bytes(c) is not None] + [f]
    return out


MODULE_REF = re.compile(r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?::[A-Za-z_]\w*)?$')


def module_closure(tree: Tree, entries, roots) -> list[str]:
    """The repository modules a spec's code loads: starting from `entries` (repository .py files: an entry point,
    a parser plugin), every `import` / `from ... import` and every string naming a module (`pkg.mod:Class`, as
    vLLM's ModelRegistry.register_model takes) that resolves to a tracked file under one of `roots` (repository
    dirs on the engine's sys.path: PYTHONPATH entries in the repository, the entry's own directory), recursively."""
    seen, todo = set(), [e for e in entries if e]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        text = tree.read_text(rel)
        if text is None:
            continue
        seen.add(rel)
        try:
            mod = ast.parse(text)
        except SyntaxError:
            continue
        names = []
        for node in ast.walk(mod):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names += [node.module] + [f'{node.module}.{a.name}' for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level:      # relative: the package of `rel`
                base = Path(rel).parent
                for _ in range(node.level - 1):
                    base = base.parent
                mods = [f'{node.module}'] if node.module else [a.name for a in node.names]
                todo += [str(base / (m.replace('.', '/') + '.py')) for m in mods]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_REF.match(node.value):
                names.append(node.value.split(':')[0])
        for name in names:
            path = name.replace('.', '/')
            for root in [*roots, str(Path(rel).parent)]:
                pre = '' if root in ('', '.') else root.rstrip('/') + '/'
                for cand in (f'{pre}{path}.py', f'{pre}{path}/__init__.py'):
                    if tree.read_bytes(cand) is not None:
                        todo.append(cand)
                        parts = cand.split('/')                    # parent packages' __init__.py run too
                        for i in range(len(pre.split('/')) if pre else 1, len(parts)):
                            todo.append('/'.join(parts[:i] + ['__init__.py']))
    return sorted(seen)


def literal_assignments(source: str, names, path: str) -> dict:
    """Module-level `NAME = <literal>` values and `NAME.update(...)` of PROFILES-style dicts, evaluated with no
    builtins and nothing but the other collected names in scope (never imports the module)."""
    tree = ast.parse(source, path)
    ns, lines = {'__builtins__': {}}, {}
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            targets = [node.targets[0].id]
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(
                node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name):
            if node.value.func.value.id in names and node.value.func.attr == 'update':
                targets = ['<update>']
        if not targets or (targets[0] != '<update>' and targets[0] not in names):
            continue
        code = compile(ast.Module(body=[node], type_ignores=[]), path, 'exec')
        try:
            exec(code, ns)   # noqa: S102  (literals and the names collected above only; no builtins)
        except Exception as e:  # noqa: BLE001
            raise ExtractError(f'{path}:{node.lineno}: cannot evaluate {targets[0]} statically ({e!r})') from e
        if targets[0] != '<update>':
            lines[targets[0]] = node.lineno
    return {**{k: v for k, v in ns.items() if k != '__builtins__'}, '__lines__': lines}


def dict_key_lines_nested(source: str, var: str, key: str) -> dict:
    """{key: line} of the literal dict at `var[key]` (a PIPELINE_PROFILES set), {} when there is none."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == var and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == key and isinstance(v, ast.Dict):
                    return {kk.value: kk.lineno for kk in v.keys if isinstance(kk, ast.Constant)}
    return {}


def dict_key_lines(source: str, var: str) -> dict:
    """{key: line} of the literal dict assigned to `var` (the PROFILES entries)."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == var and isinstance(node.value, ast.Dict):
            return {k.value: k.lineno for k in node.value.keys if isinstance(k, ast.Constant)}
    return {}


# ---------------------------------------------------------------------------------------------------------------------
# aa-serving / task-synthesis: synth/teachers/*.toml under serve_teacher.py
# ---------------------------------------------------------------------------------------------------------------------
TEACHERS = 'synth/teachers'
CAMPAIGNS = 'synth/campaigns'
SERVE_TEACHER = 'task_synthesis/serve/serve_teacher.py'
SYNTH_CONFIG = 'synth/config.py'
REPO = '@@REPO@@'     # {repo} while rendering (str.format must not see it); '{repo}' in the record
# Small checkpoint files that pin what the weights mean (as aa-serving synth/numerics.py CHECKPOINT_FILES).
CHECKPOINT_FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json', 'chat_template.jinja',
                    'chat_template.json', 'preprocessor_config.json', 'model.safetensors.index.json', 'crc32.txt')
WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt')


def deep_merge(base, override):
    """synth/config.py deep_merge."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand(value, checkpoint):
    """synth.config.load_teacher's expansion, with {repo} kept as a placeholder (the record must not depend on
    where the checkout lives)."""
    if isinstance(value, str):
        return value.replace('{checkpoint}', checkpoint).replace('{repo}', REPO)
    if isinstance(value, list):
        return [_expand(v, checkpoint) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, checkpoint) for k, v in value.items()}
    return value


def _replica_command_fallback(teacher, protocol, *, port, model_dir, tunables, rank_ports=None):
    """synth/vllm_launch.replica_command as of aa-serving 7fe25024 (used only if a branch has no vllm_launch.py)."""
    serving, runtime = teacher['serving'], teacher['runtime']
    values = {'venv': runtime['venv'], 'model_dir': model_dir, 'served_name': teacher['served_name'], 'port': port,
              'seed': serving.get('seed', 0), 'max_model_len': protocol['context_tokens'],
              'max_output_tokens': protocol['max_output_tokens'], 'checkpoint': teacher['checkpoint'],
              **(rank_ports or {}), **serving.get('tunables', {}), **tunables}
    argv = [x.format(**values) for x in serving['entrypoint']] + [x.format(**values) for x in serving['args']]
    for key, extra in serving.get('optional_args', {}).items():
        if tunables.get(key):
            argv += [x.format(**values) for x in extra]
    return argv + [str(x) for x in tunables.get('extra_args') or []]


def _replica_env_fallback(teacher, tunables):
    runtime = teacher.get('runtime', {})
    return {**runtime.get('env', {}), **{k: str(v).format(**tunables) for k, v in teacher['serving'].get('env', {}).items()},
            **(tunables.get('extra_env') or {})}


STUB_PACKAGE = '_tbshared_numerics_synth'


class Launcher:
    """A branch's own synth/vllm_launch.py (replica_command, replica_env), or the built-in copies."""

    def __init__(self, tree: Tree, pipeline: str):
        self.mod = tree.module('synth/vllm_launch.py', f'{pipeline}_vllm_launch')
        self.warnings = [] if self.mod else ['synth/vllm_launch.py missing: rendered with built-in copies']

    def command(self, teacher, protocol, tunables, rank):
        fn = getattr(self.mod, 'replica_command', None) or _replica_command_fallback
        return [a.replace(REPO, '{repo}') for a in fn(teacher, protocol, port='{port}', model_dir='{model_dir}',
                                                      tunables=tunables, rank_ports=rank)]

    def env(self, teacher, tunables) -> dict:
        """The environment replica_env sets on an empty base (what the launcher adds), with the cu13-wheel CUDA_HOME
        stubbed (it links a venv's wheels into scratch) and the scratch dir abstracted."""
        fn = getattr(self.mod, 'replica_env', None)
        if fn is None:
            return _replica_env_fallback(teacher, tunables)
        real = getattr(self.mod, 'cuda_home_from_wheel', None)
        # replica_env's lazy `from .gpu import indices` (UUIDs -> nvidia-smi indices) resolves to a stub package:
        # device numbering is job-specific (CUDA_VISIBLE_DEVICES is not recorded)
        pkg, gpu = type(sys)(STUB_PACKAGE), type(sys)(f'{STUB_PACKAGE}.gpu')
        pkg.__path__, gpu.indices = [], lambda gpus: [str(i) for i, _ in enumerate(gpus)]
        pkg.__spec__ = importlib.machinery.ModuleSpec(STUB_PACKAGE, None, is_package=True)
        gpu.__spec__ = importlib.machinery.ModuleSpec(f'{STUB_PACKAGE}.gpu', None)
        saved = {k: sys.modules.get(k) for k in (STUB_PACKAGE, f'{STUB_PACKAGE}.gpu')}
        with tempfile.TemporaryDirectory(prefix='tbshared-numerics-env-') as d:
            if real is not None:
                self.mod.cuda_home_from_wheel = lambda venvs, target: (Path(target), [])
            self.mod.__package__ = STUB_PACKAGE
            self.mod.__spec__ = importlib.machinery.ModuleSpec(f'{STUB_PACKAGE}.vllm_launch', None)
            sys.modules[STUB_PACKAGE], sys.modules[f'{STUB_PACKAGE}.gpu'] = pkg, gpu
            try:
                env = fn(teacher, gpus=['GPU-0'], scratch=d, tunables=tunables, base_env={})
            finally:
                if real is not None:
                    self.mod.cuda_home_from_wheel = real
                for k, v in saved.items():
                    sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
            return {k: str(v).replace(d, '{scratch}').replace(REPO, '{repo}') for k, v in env.items()}


def node_sha256(path) -> str:
    """sha256 of a small node file (a template on /data), or UNREADABLE...."""
    return file_identity(path)


VENV_PACKAGES = ('vllm', 'torch', 'triton', 'transformers', 'flashinfer*')


def venv_build(venv) -> dict:
    """The venv's content identity: the dist-info names of vLLM and the packages whose kernels it runs."""
    if not venv:
        return None
    site = sorted(Path(venv).glob('lib/python3*/site-packages'))
    names = sorted(p.name[:-len('.dist-info')] for s in site for pat in VENV_PACKAGES
                   for p in s.glob(f'{pat}-*.dist-info'))
    if names:
        return names
    if not Path(venv).is_dir() and host_mount_readable(Path(venv)):
        return f'missing: no venv {venv!r} on a readable {mount_of(Path(venv))}'    # a wrong path fails
    return f'{UNREADABLE}: no dist-info in {venv}'


def mount_of(p: Path) -> str:
    """The mount point a node path lives on: symlinks resolved (here /data -> /shared/data/fast, and venvs linked
    into /data), then up to the nearest real mount (os.path.ismount). A path whose filesystem is not mounted walks
    up to / (unreadable, skipped), never to an always-listable parent such as /lustre."""
    q = os.path.realpath(p)
    while q != '/' and not os.path.ismount(q):
        q = os.path.dirname(q)
    return q


def host_mount_readable(p: Path) -> bool:
    """Whether this node can read the filesystem `p` is on: its top-level directory lists. A path on a missing or
    unreadable mount is UNREADABLE (skipped); a missing path on a readable one is a real change."""
    try:
        top = Path(mount_of(p))
        return top != Path('/') and top.is_dir() and next(iter(top.iterdir()), None) is not None
    except OSError:
        return False


def checkpoint_identity(path, prefix='weights') -> dict:
    """host.<prefix>.*: the small files that pin what the weights mean (sha256), every weight shard as name:size,
    and one sha256 over the shards' file_identity samples, so a dir holding only copied configs, another SFT step
    or a re-quantized export all differ. {host.<prefix>: UNREADABLE...} when the directory cannot be read."""
    p = Path(path) if path else None
    try:
        if p is None or not p.is_dir():
            if p is not None and host_mount_readable(p):
                # the node can read this filesystem, so the directory is really missing: a wrong path must fail
                return {f'{HOST}{prefix}': f'missing: no directory {path!r} on a readable {mount_of(p)}'}
            return {f'{HOST}{prefix}': f'{UNREADABLE}: no directory {path!r}'}
        shards = sorted(f for f in p.rglob('*') if f.is_file() and f.suffix in WEIGHT_SUFFIXES
                        and not any(part.startswith('.') for part in f.relative_to(p).parts))
        out = {f'{HOST}{prefix}.files_sha256.{name}': file_identity(p / name) for name in CHECKPOINT_FILES
               if (p / name).is_file()}
        out[f'{HOST}{prefix}.shards'] = [f'{f.relative_to(p)}:{f.stat().st_size}' for f in shards]
        with concurrent.futures.ThreadPoolExecutor(16) as pool:      # /data latency, not bandwidth, dominates
            samples = list(pool.map(file_identity, shards))
        out[f'{HOST}{prefix}.shards.sample_sha256'] = hashlib.sha256(json.dumps(samples).encode()).hexdigest()
    except OSError as e:
        return {f'{HOST}{prefix}': f'{UNREADABLE}: {type(e).__name__}'}
    return out


def render(tree: Tree, launcher: Launcher, num, teacher: dict, protocol: dict, tunables: dict, rank_extra=None,
           honours_cuda_capability=True):
    """The fields of one engine of a (merged, expanded) teacher spec under `protocol` and `tunables`."""
    serving, runtime = teacher['serving'], teacher.get('runtime', {})
    ckpt = teacher.get('checkpoint') or ''
    rank = {'dp_size': '{dp_size}', 'rpc_port': '{rpc_port}', 'supervisor_port': '{supervisor_port}',
            **(rank_extra or {})}
    argv = launcher.command(teacher, protocol, tunables, rank)
    venv = tunables.get('venv') or runtime.get('venv')
    if ckpt:   # the checkpoint's location is not numerics; its files are (host.weights.*)
        argv = [a.replace(ckpt, '{checkpoint}') for a in argv]
    fields, warn = argv_fields(argv)
    entry = len(serving.get('entrypoint', []))
    fields.update({
        'weights.model': teacher.get('model'), 'weights.revision': teacher.get('revision'),
        'weights.declared_quantization': teacher.get('quantization'),
        'weights.quantization': _quant_label(num, argv),
        'weights.checkpoint_name': Path(ckpt).name if ckpt else None,   # repo-derived: a renamed model fails anywhere
        **checkpoint_identity(ckpt),
        # a profile that pins another venv (tunables venv) declares its version with it (tunables vllm_version)
        'vllm.version': tunables.get('vllm_version') or runtime.get('vllm_version'), f'{HOST}vllm.build': venv_build(venv),
        'vllm.cuda_home': runtime.get('cuda_home'),
        'vllm.entrypoint': [a.replace(str(venv), '{venv}') for a in argv[:entry]],
        'gpu.cuda_capability': (serving_resources_cc(teacher) if honours_cuda_capability else '10.0'),
    })
    tpl = fields['chat_template.path']
    if tpl.startswith('{checkpoint}/'):
        fields[f'{HOST}chat_template.sha256'] = node_sha256(Path(ckpt) / tpl[len('{checkpoint}/'):])
    elif tpl.startswith('/'):
        fields[f'{HOST}chat_template.sha256'] = node_sha256(tpl)
    if teacher.get('tokenizer_dir'):
        fields.update(checkpoint_identity(teacher['tokenizer_dir'], 'weights.tokenizer_dir'))
    for key in ('cu13_fallback_venvs', 'extra_ld_library_paths', 'nvcc_append_flags', 'cu13_ld_library_path'):
        if runtime.get(key):
            v = runtime[key]
            if key == 'cu13_fallback_venvs':
                fields[f'vllm.{key}'] = len(v)
                fields[f'{HOST}vllm.{key}'] = [venv_build(x) for x in v]
            else:
                fields[f'vllm.{key}'] = v
    if serving.get('kernel_wheels'):
        fields['vllm.kernel_wheels'] = sorted([Path(w.get('path', '')).name, w.get('sha256')]
                                              for w in serving['kernel_wheels'])
    for x in serving.get('stage_extra', []):
        fields[f'weights.stage_extra.{x.get("name")}'] = 'staged'
        fields.update(checkpoint_identity(x.get('source', ''), f'weights.stage_extra.{x.get("name")}'))
    env = launcher.env(teacher, tunables)
    fields.update(env_fields(env, skip=set(runtime.get('cache_env', []))))
    roots = [p[len('{repo}/'):] if p != '{repo}' else '' for p in env.get('PYTHONPATH', '').split(':')
             if p == '{repo}' or p.startswith('{repo}/')]
    entries = [a[len('{repo}/'):] for a in fields['vllm.entrypoint'] if a.startswith('{repo}/') and a.endswith('.py')]
    entries += [str(v)[len('{repo}/'):] for k, v in fields.items() if k.endswith('parser_plugin')
                and str(v).startswith('{repo}/')]
    entries += entry_point_modules(tree, [r for r in roots if r])
    if entries:
        mods = module_closure(tree, entries, roots)
        fields['model_impl.repo_modules'] = mods
        fields['model_impl.repo_modules.sha256'] = _tree_digest(tree, mods)
    add_repo_hashes(fields, tree, '{repo}/')
    request = teacher.get('request', {})
    fields['request.pi_thinking'] = request.get('pi_thinking')
    flatten('request.overrides', request.get('overrides') or {}, fields)
    return fields, warn


def _reads_key(source: str, key: str) -> bool:
    """Whether code reads `key` from a dict (`.get('key'` or `['key']`), not merely mentions it."""
    return re.search(r"(\.get\(\s*|\[)['\"]" + re.escape(key) + r"['\"]", source) is not None


def serving_resources_cc(teacher) -> str:
    """The GPU type a serve_teacher job requests (serving_resources.cuda_capability; serve_teacher's default 10.0)."""
    return str((teacher.get('serving_resources') or {}).get('cuda_capability', '10.0'))


def toml_specs(tree: Tree, pipeline: str) -> dict:
    """{specs, sources, skipped, warnings} for a serve_teacher pipeline: every synth/teachers/*.toml under
    serve_teacher's PROFILES (one spec per replica arm when a profile sets replica_tunables), and every
    synth/campaigns/*.toml as synth/serve.py runs it (campaign/<name>) with its tuning candidates
    (campaign/<name>/candidate/<candidate>)."""
    out = {'specs': {}, 'sources': {}, 'skipped': [], 'warnings': []}
    if tomllib is None:
        out['skipped'].append(f'all {TEACHERS}/*.toml: Python {sys.version_info[0]}.{sys.version_info[1]} has no '
                              'tomllib (3.11+); run the guard with Python 3.11 or newer')
        return out
    st_src = tree.read_text(SERVE_TEACHER)
    if st_src is None:
        raise ExtractError(f'{tree.label}: no {SERVE_TEACHER}')
    st = literal_assignments(st_src, {'PROTOCOL', 'PROFILES', 'TEXT_ONLY_VARIANTS', 'PIPELINE_PROFILES'}, SERVE_TEACHER)
    if 'PROTOCOL' not in st or 'PROFILES' not in st:
        raise ExtractError(f'{SERVE_TEACHER}: PROTOCOL / PROFILES not found as module-level literals')
    # the trunk's one launcher (#480): a pipeline's own complete profiles (serve_teacher --pipeline) replace PROFILES
    # entries for the teachers it names; a branch launcher without PIPELINE_PROFILES serves only its own pipeline
    own = (st.get('PIPELINE_PROFILES') or {}).get(pipeline, {})
    st['profiles'] = {**st['PROFILES'], **own}
    st['pipeline_profiles'] = set(own)
    # whether the launcher stages serving.stage_extra and passes {stage_<name>} (synth.serve.stage_teacher, #445)
    st['stages_extra'] = _calls(st_src, 'stage_teacher')
    st['replica_tunables'] = _reads_key(st_src, 'replica_tunables')   # aa-serving's per-replica arms (#110)
    st['cuda_capability'] = _reads_key(st_src, 'cuda_capability')     # the spec's GPU type (else 10.0)
    profile_lines = dict_key_lines(st_src, 'PROFILES')
    pipeline_lines = dict_key_lines_nested(st_src, 'PIPELINE_PROFILES', pipeline)
    launcher = Launcher(tree, pipeline)
    out['warnings'] += launcher.warnings
    num = tree.module('synth/numerics.py', f'{pipeline}_synth_numerics')
    teachers = {}
    for rel in tree.list(TEACHERS):
        if not rel.endswith('.toml'):
            continue
        slug = Path(rel).stem
        try:
            spec = tomllib.loads(tree.read_text(rel))
            if spec.get('slug') != slug:
                raise ExtractError(f'declares slug {spec.get("slug")!r}')
            teachers[slug] = spec
            arms = teacher_spec(tree, slug, spec, st, launcher, num)
        except Exception as e:  # noqa: BLE001  (one bad spec must not hide the others)
            out['skipped'].append(_skip(tree, rel, e))
            continue
        if slug in st['pipeline_profiles']:
            line = pipeline_lines.get(slug)
            prof = (f'{SERVE_TEACHER}:{line}' if line else SERVE_TEACHER) + f' (PIPELINE_PROFILES[{pipeline!r}])'
        elif slug in st['PROFILES']:
            line = profile_lines.get(slug) or profile_lines.get(slug[:-len('-text')] if slug.endswith('-text') else '')
            prof = f'{SERVE_TEACHER}:{line}' if line else f'{SERVE_TEACHER} (PROFILES, derived entry)'
        else:
            prof = f'{SERVE_TEACHER}:{st["__lines__"].get("PROTOCOL")} (PROTOCOL; no PROFILES entry)'
        for key, (fields, warn) in arms.items():
            out['specs'][key], out['sources'][key] = fields, {'spec': rel, 'profile': prof}
            out['warnings'] += [f'{key}: {w}' for w in warn]
    campaign_specs(tree, teachers, launcher, num, out)
    return out


def teacher_spec(tree, slug, spec, st, launcher, num) -> dict:
    """{key: (fields, warnings)} of one teacher under serve_teacher: `slug`, or `slug@r<i>` per replica arm."""
    teacher = _expand(spec, spec['checkpoint'])
    serving = teacher['serving']
    profile = st['profiles'].get(slug, {})
    protocol = profile.get('protocol', st['PROTOCOL'])
    tunables = {**serving.get('tunables', {}), **profile.get('tunables', {})}
    warn, rank = [], {}
    staged = {f'stage_{x["name"]}': f'{{stage_{x["name"]}}}' for x in serving.get('stage_extra', [])}
    if staged:   # synth/serve.py stages these (draft models) and passes their paths; old serve_teachers did not
        if not st.get('stages_extra'):
            warn.append(f'stage_extra {sorted(staged)}: serve_teacher passes no such values (synth/serve.py does), so '
                        'this spec fails to launch under serve_teacher; rendered with placeholders')
        rank.update(staged)
    arms = (tunables.get('replica_tunables') or []) if st['replica_tunables'] else []
    out = {}
    for i, arm in enumerate(arms or [{}]):
        tun = {**tunables, **arm}
        fields, w = render(tree, launcher, num, teacher, protocol, tun, rank, st['cuda_capability'])
        out[f'{slug}@r{i}' if arms else slug] = (fields, warn + w)
    return out


def campaign_specs(tree, teachers, launcher, num, out):
    """Campaigns as synth/serve.py serves them: config.build's merged teacher (teacher_overrides deep-merged),
    protocol (PROTOCOL_DEFAULTS < teacher [protocol] < campaign [protocol]) and the teacher's tunables (the control
    file's defaults); a tuning candidate deep-merges its own teacher overrides and tunables on top (synth/tune.py)."""
    names = [r for r in tree.list(CAMPAIGNS) if r.endswith('.toml')]
    if not names:
        return
    cfg_src = tree.read_text(SYNTH_CONFIG) or ''
    defaults = literal_assignments(cfg_src, {'PROTOCOL_DEFAULTS'}, SYNTH_CONFIG).get('PROTOCOL_DEFAULTS') if cfg_src else None
    if not defaults:
        out['skipped'].append(f'{CAMPAIGNS}/*.toml: no PROTOCOL_DEFAULTS literal in {SYNTH_CONFIG}')
        return
    code = ''.join(tree.read_text(f) or '' for f in ('synth/serve.py', 'synth/condor.py', 'synth/controller.py'))
    honours_cc = _reads_key(code, 'cuda_capability')
    with _tb_agents(tree):
        notice = synth_finish_notice(cfg_src)
        for rel in names:
            _campaign(tree, rel, teachers, launcher, num, out, defaults, honours_cc, notice)


def synth_finish_notice(cfg_src: str):
    """The branch's synth/config.finish_notice_config (with its FINISH_NOTICE_DEFAULT / FINISH_OFF), run against the
    branch's own tb_agents.pi_context_limit (see _tb_agents), so a campaign's notice is recorded as the frozen record
    config.build writes; None when the branch has no such function (the raw setting is recorded then)."""
    tree = ast.parse(cfg_src)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'finish_notice_config'), None)
    if fn is None:
        return None
    ns = {'__builtins__': builtins, 'REPO': '{repo}',
          **{k: v for k, v in literal_assignments(cfg_src, {'FINISH_NOTICE_DEFAULT', 'FINISH_OFF'}, SYNTH_CONFIG).items()
             if k != '__lines__'}}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), SYNTH_CONFIG, 'exec'), ns)   # noqa: S102  (the branch's code)
    return ns['finish_notice_config']


def _skip(tree, rel, e) -> str:
    """A skip entry that still changes when the file does (its sha256)."""
    return f'{rel} (sha256 {tree.sha256(rel)}): {e!r}'


TEMPLATE_MARKER = re.compile(r'@[A-Z_]+@')
# TOML strings (multi-line first; basic strings with escapes) and comments: markers inside them are left alone
_TOML_SKIP = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'[^\'\n]*\'|#[^\n]*')


def _calls(source: str, name: str) -> bool:
    """Whether `source` calls a function named `name` (a bare name or an attribute), by its AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Call) and (getattr(n.func, 'id', None) == name or getattr(n.func, 'attr', None) == name)
               for n in ast.walk(tree))


def fill_template(text: str) -> str:
    """A campaign template (task_synthesis/pipeline/production.py campaign_toml fills its @MARKER@s: run id, stream dir,
    smoke attempts, budgets, engine/worker counts, bids, autoscale) as parseable TOML: a line that is only a marker is
    dropped, a marker inside a string or comment stays, any other becomes 0. Markers name per-run and throughput
    settings only, so the serving numerics are the template's own (#445: it was skipped as invalid TOML)."""
    text = '\n'.join(ln for ln in text.splitlines() if not TEMPLATE_MARKER.fullmatch(ln.strip())) + '\n'
    out, pos = [], 0
    for m in _TOML_SKIP.finditer(text):
        out += [TEMPLATE_MARKER.sub('0', text[pos:m.start()]), m.group(0)]
        pos = m.end()
    return ''.join(out) + TEMPLATE_MARKER.sub('0', text[pos:])


def _campaign(tree, rel, teachers, launcher, num, out, defaults, honours_cc, notice):
    """One campaign (and its tuning candidates) into `out`; a template (@MARKER@s) through fill_template."""
    key = f'campaign/{Path(rel).stem}'
    try:
        text = tree.read_text(rel)
        template = bool(TEMPLATE_MARKER.search(text))
        raw = tomllib.loads(fill_template(text) if template else text)
        slug = raw.get('teacher')
        if slug not in teachers:
            raise ExtractError(f'teacher {slug!r} is not a {TEACHERS}/*.toml spec')
        merged = deep_merge(teachers[slug], raw.get('teacher_overrides'))
        teacher = _expand(merged, merged['checkpoint'])
        teacher['serving_resources'] = deep_merge(teacher.get('serving_resources', {}), raw.get('serving', {}))
        protocol = {**defaults, **teacher.get('protocol', {}), **raw.get('protocol', {})}
        staged = {f'stage_{x["name"]}': f'{{stage_{x["name"]}}}' for x in teacher['serving'].get('stage_extra', [])}
        cands = [('', teacher, teacher['serving'].get('tunables', {}))]
        for c in (raw.get('tuning') or {}).get('candidates', []):
            t = deep_merge(teacher, c.get('teacher', {}))
            cands.append((f'/candidate/{c.get("name")}', t, {**t['serving'].get('tunables', {}),
                                                             **c.get('tunables', {})}))
        if notice is not None:    # synth/config.check_protocol: the campaign's and every override's notice
            protocol['finish_notice'] = notice(protocol.get('finish_notice'))
            protocol['overrides'] = [{**o, **({'finish_notice': notice(o['finish_notice'])}
                                              if 'finish_notice' in o else {})}
                                     for o in protocol.get('overrides') or []]
        for suffix, t, tun in cands:
            fields, warn = render(tree, launcher, num, t, protocol, tun, staged, honours_cc)
            fields['campaign.teacher'] = slug
            if template:
                fields['campaign.template'] = True
            for k in ('context_limit_action', 'compaction', 'pi_version'):
                if k in protocol:
                    fields[f'protocol.{k}'] = protocol[k]
            if 'finish_notice' in protocol:
                flatten('protocol.finish_notice', protocol['finish_notice'], fields)
            if protocol.get('overrides'):
                fields['protocol.overrides'] = protocol['overrides']
            out['specs'][key + suffix] = fields
            out['sources'][key + suffix] = {'campaign': rel, 'teacher': f'{TEACHERS}/{slug}.toml'}
            out['warnings'] += [f'{key + suffix}: {w}' for w in warn]
    except Exception as e:  # noqa: BLE001
        out['skipped'].append(_skip(tree, rel, e))


def _quant_label(num, argv, model_dir='{model_dir}'):
    """The weight-quantization label as the pipeline's own runtime record computes it (aa-serving
    synth/numerics.py, eval_q35/serve_numerics.py `weight_quantization`), from argv alone: the checkpoint's
    config.json is not read here, so a checkpoint-native quantization shows as 'checkpoint dtype' and the spec's
    declared label (weights.declared_quantization) says what it is. A branch without that module gets the same
    rule from here."""
    fn = getattr(num, 'weight_quantization', None)
    if fn is not None:
        try:
            return fn(argv, model_dir)['label']
        except Exception as e:  # noqa: BLE001
            return f'unavailable ({type(e).__name__})'
    flags = parse_argv(argv)
    online = flags.get('--quantization') or None
    injected = (_json(flags.get('--hf-overrides')) or {}).get('quantization_config') if flags.get(
        '--hf-overrides') else None
    if isinstance(injected, dict) and injected:
        label = '/'.join(str(x) for x in (injected.get('quant_method'), injected.get('quant_algo')
                                          or injected.get('fmt')) if x)
        return label + (f' + online {online}' if online and online != injected.get('quant_method') else '')
    return f'{online} (online, from checkpoint dtype)' if online else 'checkpoint dtype'


# ---------------------------------------------------------------------------------------------------------------------
# eval: eval_q35/submit.py init + eval_q35/serve_job.sh
# ---------------------------------------------------------------------------------------------------------------------
EVAL = 'eval_q35'
# Repository templates a family's runs use by default (submit.py docstring: base Qwen3.5 models get
# --chat-template sft_qwen35/chat_template.jinja), recorded with their sha256.
EVAL_FAMILY_TEMPLATES = {'qwen3_5': ['sft_qwen35/chat_template.jinja']}
GPU_CC = {'h100': '9.0', 'h100x2': '9.0', 'b200': '10.0', 'b200x2': '10.0'}


class _FakePath:
    """A run directory that records nothing and touches no disk (submit.py init's `run`)."""

    def __init__(self, s='{run}'):
        self.s = s

    def __truediv__(self, other):
        return _FakePath(f'{self.s}/{other}')

    def __str__(self):
        return self.s

    def __fspath__(self):
        return self.s

    def exists(self):
        return False

    def mkdir(self, *a, **k):
        pass

    def write_text(self, *a, **k):
        pass

    @property
    def name(self):
        return self.s.rsplit('/', 1)[-1]


SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'float', 'int', 'isinstance', 'len', 'list', 'max', 'min',
    'range', 'set', 'sorted', 'str', 'sum', 'tuple', 'zip', 'Exception', 'SystemExit', 'ValueError', 'KeyError',
    'TypeError', 'AssertionError', 'getattr', 'hasattr', 'repr', 'frozenset', 'round', 'type', 'iter', 'next',
    'reversed', 'map', 'filter', 'divmod', 'callable', 'NotImplementedError', 'OSError', 'FileNotFoundError')}


class _ArgvParse(ast.NodeTransformer):
    """`X.parse_args()` -> `X.parse_args(_ARGV)`."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'parse_args' and not node.args:
            node.args = [ast.Name(id='_ARGV', ctx=ast.Load())]
        return node


def eval_config(submit_src: str, common, argv: list[str], family: str = '') -> dict:
    """The config.json that `submit.py ARGV` (action init) would write: main()'s argument parser and its `init`
    block, executed in restricted globals with stubbed I/O: no git snapshot, no files written, nothing but the
    stubs, common.py's names, argparse and a few safe builtins in scope (no open, no imports, no subprocess)."""
    tree = ast.parse(submit_src, f'{EVAL}/submit.py')
    main = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main'), None)
    if main is None:
        raise ExtractError(f'{EVAL}/submit.py: no main()')
    pre, init = [], None
    for node in main.body:
        test = ast.unparse(node.test) if isinstance(node, ast.If) else ''
        if 'a.action' in test and "'init'" in test:
            init = node
            break
        pre.append(node)
    if init is None:
        raise ExtractError(f"{EVAL}/submit.py main(): no `if a.action == 'init'` block")
    captured = {}

    def atomic(path, value):
        captured[str(path).rsplit('/', 1)[-1]] = copy.deepcopy(value)

    allow = getattr(common, 'GRUG_TEMPLATES', {}).get(family)
    ns = {'__builtins__': {**SAFE_BUILTINS, 'print': lambda *a, **k: None},
          **{k: getattr(common, k) for k in dir(common) if not k.startswith('_')},
          'argparse': argparse, 'Path': Path, 'json': json, 're': re}
    stubs = {'atomic': atomic, 'read': lambda *a, **k: (a[1] if len(a) > 1 else k.get('default')),
             # a family pinned to template hashes gets an allowed one (the per-run template is a placeholder)
             'sha256': lambda p: sorted(allow)[0] if allow else '{sha256 of ' + str(p) + '}',
             'now': lambda: '{now}', 'tasks': lambda *a, **k: ['{task}'], 'snapshot': lambda: ('{commit}', '{code}'),
             'ROOT': _FakePath('{root}'), '_ARGV': list(argv),
             # the harness of a checkpoint without SFT provenance (a placeholder model has none): --harness or pi
             'harness_for': lambda a, model: {'harness': a.harness or 'pi',
                                              'harness_source': 'explicit' if a.harness else 'default'}}
    for node in tree.body:                 # submit.py's own helpers and constants (definitions only: nothing runs)
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            exec(compile(ast.Module(body=[node], type_ignores=[]), f'{EVAL}/submit.py', 'exec'), ns)   # noqa: S102
        elif isinstance(node, ast.Assign):
            with contextlib.suppress(Exception):
                exec(compile(ast.Module(body=[node], type_ignores=[]), f'{EVAL}/submit.py', 'exec'), ns)   # noqa: S102
    ns.update(stubs)
    fn = ast.parse('def _tbshared_init():\n    pass').body[0]
    fn.body = [_ArgvParse().visit(copy.deepcopy(n)) for n in pre] + init.body + [ast.Return(value=ast.Constant(None))]
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(mod, f'{EVAL}/submit.py', 'exec'), ns)   # noqa: S102  (restricted globals: see the docstring)
    try:
        ns['_tbshared_init']()
    except SystemExit as e:
        raise ExtractError(f'submit.py {" ".join(argv)}: exited ({e})') from None
    except Exception as e:  # noqa: BLE001
        raise ExtractError(f'submit.py {" ".join(argv)}: {e!r} (the stubbed init could not run)') from e
    if 'config.json' not in captured:
        raise ExtractError(f'submit.py {" ".join(argv)}: wrote no config.json')
    return captured['config.json']


# --- serve_job.sh: a small interpreter for the statements that build the vLLM command ---------------------------------
ARRAYS = ('FAMILY_FLAGS', 'QUANT_FLAGS', 'TOOL_FLAGS', 'SERVE_ENV', 'LAUNCH', 'VARGS')
PINNED = {'RUN': '{run}', 'CODE': '{code}', 'SCR': '{scratch}', 'M': '{model_dir}', 'PORT': '{port}', 'HOST': '{host}',
          'TAG': '{tag}', 'LOGS': '{run}/serve', 'PY': '{harbor_python}'}   # job-specific: placeholders


def _step(s: str, i: int, stack: list) -> int:
    """Advance over s[i] (or a two-character `$(` / `<(`), updating the quote/substitution stack; returns the next
    index. Stack entries: "'", '"', '(' (a substitution, subshell or array)."""
    c, top = s[i], (stack[-1] if stack else None)
    if top == "'":
        if c == "'":
            stack.pop()
        return i + 1
    if top == '"':
        if c == '\\':
            return i + 2
        if c == '"':
            stack.pop()
        elif s.startswith('$(', i):
            stack.append('(')
            return i + 2
        return i + 1
    if c == '\\':
        return i + 2
    if c in '\'"':
        stack.append(c)
    elif c == '(':
        stack.append('(')
    elif c == ')' and top == '(':
        stack.pop()
    return i + 1


def _split_statements(text: str):
    """(line, statement) pairs: comments and heredoc bodies dropped, continued lines joined (open quotes or
    parentheses, a trailing `\\`, `|`, `&&` or `||`), and `;`-separated statements split at top level (a `while`
    statement stays whole)."""
    out, buf, start, stack, heredoc = [], '', None, [], None
    for n, line in enumerate(text.splitlines(), 1):
        if heredoc:
            if line.strip() == heredoc:
                heredoc = None
            continue
        i, cut = 0, len(line)
        while i < len(line):
            if not stack and line[i] == '#' and (i == 0 or line[i - 1] in ' \t;'):
                cut = i
                break
            i = _step(line, i, stack)
        cleaned = line[:cut].strip()
        m = re.search(r"<<-?\s*'?([A-Za-z_]+)'?", cleaned) if not stack else None
        if m:
            heredoc = m.group(1)
        if start is None:
            start = n
        buf = f'{buf} {cleaned}'.strip() if cleaned else buf
        if stack or buf.endswith(('\\', '|', '&&', '||')):
            buf = buf[:-1].rstrip() if buf.endswith('\\') else buf
            continue
        if buf:
            for part in ([buf] if buf.startswith('while ') else _top_split(buf)):
                out.append((start, part))
        buf, start = '', None
    return out


def _top_split(s: str):
    parts, cur, stack, i = [], 0, [], 0
    while i < len(s):
        if not stack and s[i] == ';':
            parts.append(s[cur:i].strip())
            cur = i + 1
            i += 1
            continue
        i = _step(s, i, stack)
    parts.append(s[cur:].strip())
    return [p for p in parts if p and p != ';']


def _words(s: str):
    """Shell words of `s` as (text, quoted), quotes and substitutions kept verbatim inside each word."""
    words, cur, stack, i, quoted = [], '', [], 0, False
    while i < len(s):
        if not stack and s[i] in ' \t':
            if cur:
                words.append((cur, quoted))
            cur, quoted, i = '', False, i + 1
            continue
        if not stack and s[i] in '\'"':
            quoted = True
        j = _step(s, i, stack)
        cur += s[i:j]
        i = j
    if cur:
        words.append((cur, quoted))
    return words


def _close(s: str, i: int) -> int:
    """Index just past the `)` that closes the `$(` at s[i]."""
    stack, j = ['('], i + 2
    while j < len(s) and stack:
        j = _step(s, j, stack)
    return j


class ServeJob:
    """Runs the subset of serve_job.sh that builds `LAUNCH` and `VARGS` for one config.json and node GPU: variable
    assignments, `cfg` lookups, array assignments, `[ test ] &&` guards, if/elif/else, and the serve_env loop.
    A statement that touches those arrays in any other way, or an unknown command substitution that reaches them,
    is reported in `unparsed`."""

    def __init__(self, text: str, config: dict, cc: str, path=f'{EVAL}/serve_job.sh', reader=None):
        self.config, self.cc, self.path = config, cc, path
        self.reader = reader              # rel -> text of a repository file (`source "$CODE/..."`), or None
        self.vars = dict(PINNED)
        self.arrays = {}
        self.exports = {}
        self.unparsed = []
        self.writes = []                  # statements that write into the staged checkpoint or chat template
        self.depth = 0
        self.statements = _split_statements(text)

    def cfg(self, expr: str):
        expr = expr.replace("'$CC'", repr(self.cc))
        try:
            node = ast.parse('v' + expr, mode='eval')
        except SyntaxError:
            raise ExtractError(f'{self.path}: cfg expression {expr!r} is not Python') from None
        allowed = (ast.Expression, ast.Name, ast.Load, ast.Attribute, ast.Call, ast.Subscript, ast.Constant, ast.Dict,
                   ast.BoolOp, ast.Or, ast.List, ast.Tuple)
        for sub in ast.walk(node):
            if not isinstance(sub, allowed) or (isinstance(sub, ast.Attribute) and sub.attr != 'get') or (
                    isinstance(sub, ast.Name) and sub.id != 'v'):
                raise ExtractError(f'{self.path}: cfg expression {expr!r} uses more than .get / [] / or')
        v = eval(compile(node, 'cfg', 'eval'), {'__builtins__': {}}, {'v': self.config})   # noqa: S307
        return v if isinstance(v, str) else json.dumps(v)

    def subst(self, body: str, line: int) -> str:
        b = body.strip()
        m = re.fullmatch(r'cfg\s+"(.*)"', b, re.S)
        if m:
            return self.cfg(m.group(1))
        if 'nvidia-smi' in b and 'compute_cap' in b:
            return self.cc.replace('.', '') if "tr -d '.'" in b else self.cc
        m = re.fullmatch(r'\[\s((?:[^\]])*?)\s\]\s*&&\s*echo\s+"([^"]*)"\s*\|\|\s*echo\s+"([^"]*)"', b, re.S)
        if m and self.test(m.group(1), line) is not None:
            return self.expand(m.group(2) if self.test(m.group(1), line) else m.group(3), line)
        return '{$(' + b[:40] + ')}'          # unknown: reported if it reaches LAUNCH / VARGS

    def expand(self, s: str, line: int) -> str:
        """$VAR, ${VAR}, ${ARR[@]} and $(...) inside a double-quoted or unquoted string."""
        out, i = '', 0
        while i < len(s):
            if s.startswith('$(', i):
                j = _close(s, i)
                out += self.subst(s[i + 2:j - 1], line)
                i = j
                continue
            m = re.match(r'\$(?:\{([A-Za-z_]\w*)(\[@\])?\}|([A-Za-z_]\w*))', s[i:])
            if m:
                name = m.group(1) or m.group(3)
                # a variable this script never sets comes from the job's environment: unknown here
                out += ' '.join(self.arrays.get(name, [])) if m.group(2) else self.vars.get(name, '{?' + name + '}')
                i += m.end()
                continue
            m = re.match(r'\$\{[^}]*\}', s[i:])            # ${VAR%suffix} and other parameter expansions
            if m:
                out += '{?' + m.group(0)[2:-1] + '}'
                i += m.end()
                continue
            out += s[i]
            i += 1
        return out

    def expand_word(self, word: str, quoted: bool, line: int) -> list[str]:
        """One shell word -> its fields: "${ARR[@]}" -> the elements, an unquoted expansion is word-split."""
        m = re.fullmatch(r'"\$\{([A-Za-z_]\w*)\[@\]\}"', word)
        if m:
            return list(self.arrays.get(m.group(1), []))
        text, i, split = '', 0, False
        while i < len(word):
            if word[i] in '\'"':
                stack, j = [], i
                j = _step(word, j, stack)
                while j < len(word) and stack:
                    j = _step(word, j, stack)
                inner = word[i + 1:j - 1]
                text += inner if word[i] == "'" else self.expand(inner, line)
                i = j
                continue
            j, stack = i, []
            while j < len(word) and (stack or word[j] not in '\'"'):
                j = _step(word, j, stack)
            seg = word[i:j]
            split = split or '$' in seg
            text += self.expand(seg, line)
            i = j
        return text.split() if split and not quoted else [text]

    def words(self, s: str, line: int) -> list[str]:
        return [x for w, q in _words(s) for x in self.expand_word(w, q, line)]

    @staticmethod
    def unknown(text: str) -> bool:
        return '{$(' in text or '{?' in text

    def test(self, cond: str, line: int) -> bool:
        """True / False, or None when the test is not one this interpreter knows or depends on something it cannot
        know here (a command substitution it does not model, a variable from the job's environment)."""
        w = [x for wd, q in _words(cond.strip()) for x in (self.expand_word(wd, q, line) or [''])]
        if any(self.unknown(x) for x in w):
            return None
        if len(w) == 2 and w[0] in ('-n', '-z'):
            return (w[1] != '') == (w[0] == '-n')
        if len(w) == 2 and w[0] in ('-d', '-f', '-e'):
            return True                              # node paths (prebuilt kernel dirs): assumed present
        if len(w) == 3 and w[1] in ('=', '==', '!='):
            return (w[0] == w[2]) == (w[1] != '!=')
        if len(w) == 3 and w[1] in ('-gt', '-ge', '-lt', '-le', '-eq', '-ne'):
            try:
                a, b = int(w[0]), int(w[2])
            except ValueError:
                return None
            return {'-gt': a > b, '-ge': a >= b, '-lt': a < b, '-le': a <= b, '-eq': a == b, '-ne': a != b}[w[1]]
        return None                                   # not a test this interpreter knows

    def guarded(self, st: str):
        """(conditions, statement) of `[ c1 ] && [ c2 ] && statement`."""
        conds = []
        while True:
            m = re.match(r'\[\s((?:[^\]]|\[@\])*?)\s\]\s*&&\s*', st)
            if not m:
                return conds, st
            conds.append(m.group(1))
            st = st[m.end():]

    def touches(self, st: str) -> bool:
        return any(re.search(rf'\b{a}\b', st) for a in ARRAYS)

    def run(self):
        self._run(self.statements)
        for name in ('LAUNCH', 'VARGS'):
            for x in self.arrays.get(name, []):
                if self.unknown(x):
                    self.unparsed.append(f'{self.path}: {name} contains a value it cannot know here: {x[:60]}')
        for k, v in self.exports.items():
            if self.unknown(v):
                self.unparsed.append(f'{self.path}: export {k} has a value it cannot know here: {v[:60]}')
        return self

    def _run(self, statements):
        # one frame per open block: [this branch taken, some branch taken, unknown, kind]; a loop, `case` or function
        # body is unknown (this interpreter does not iterate, match or call): every variable such a block assigns or
        # exports becomes unknown ({?name}, reported wherever it reaches the launch), and a statement in it that
        # touches the arrays or sources a file is reported
        stack = []
        for line, st in statements:
            while True:
                m = re.match(r'(then|else|do)\s+(.*)', st, re.S)
                if not m:
                    break
                if m.group(1) == 'else' and stack and stack[-1][3] == 'if':
                    stack[-1] = [not stack[-1][1], True, stack[-1][2], 'if']
                st = m.group(2)
            self.check_write(line, st)
            m = re.fullmatch(r'(if|elif)\s+\[\s((?:[^\]]|\[@\])*?)\s\]', st)
            if st.startswith(('if ', 'elif ')):
                outer = stack[:-1] if st.startswith('elif ') else stack
                live = all(t for t, _, _, _ in outer) and not any(u for _, _, u, _ in outer)
                cond = self.test(m.group(2), line) if (m and live) else None
                if st.startswith('if '):
                    stack.append([bool(cond), bool(cond), live and cond is None, 'if'])
                elif stack:
                    taken = bool(cond) and not stack[-1][1]
                    stack[-1] = [taken, stack[-1][1] or taken,
                                 stack[-1][2] or (live and not stack[-1][1] and cond is None), 'if']
                continue
            m_for = re.match(r'(for|select)\s+([A-Za-z_]\w*)\s', st)
            if m_for or re.match(r'until\s', st) or (st.startswith('while ') and not re.search(r'\bdone\b', st)):
                stack.append([False, False, True, 'loop'])
                if m_for:
                    self.taint_names([m_for.group(2)])
                continue
            if st.startswith('case '):
                stack.append([False, False, True, 'case'])
                st = re.sub(r'^case\s+.*?\s+in\b\s*', '', st)
                if not st:
                    continue
            if re.match(r'(function\s+)?[A-Za-z_]\w*\s*\(\)\s*\{', st):
                stack.append([False, False, True, 'func'])
                st = re.sub(r'^(function\s+)?[A-Za-z_]\w*\s*\(\)\s*\{\s*', '', st)
            if st in ('else',) and stack and stack[-1][3] == 'if':
                stack[-1] = [not stack[-1][1], True, stack[-1][2], 'if']
                continue
            closer = {'fi': 'if', 'done': 'loop', 'esac': 'case', '}': 'func'}.get(st)
            if closer:
                if stack and stack[-1][3] == closer:
                    stack.pop()
                continue
            if st in ('then', 'do', ''):
                continue
            if any(u for _, _, u, _ in stack):
                self.unknown_statement(line, st, 'in a block it cannot follow')
                continue
            if all(t for t, _, _, _ in stack):
                self.statement(line, st)

    WRITERS = {'cp', 'mv', 'install', 'rsync', 'ln', 'tee', 'truncate', 'dd', 'sed', 'perl', 'patch', 'python',
               'python3', 'chmod', 'rm'}

    def check_write(self, line: int, st: str):
        """Report a statement that may write into what the engine loads: the staged checkpoint ($M -> {model_dir})
        or the model source / template (from config.json: {model}...), by a writing command (sed -i, cp, mv, dd of=,
        tee, ...) or a `>` / `>>` redirection. Every branch is checked, taken or not (a write is recorded once per
        script; the staging copies themselves are recorded too, so only a new or changed write fails)."""
        body = re.sub(r'^case\s+.*?\s+in\b\s*', '', self.guarded(st)[1])
        body = re.sub(r'^\(?[^\s()=]+\)\s*', '', body) if re.match(r'^\(?[^\s()=]+\)', body) else body
        body = self.guarded(body)[1]
        try:
            words = [(w, x) for w, q in _words(body) for x in [' '.join(self.expand_word(w, q, line))]]
        except Exception:  # noqa: BLE001  (best effort: this check must not stop the extraction)
            return
        staged = lambda x: '{model_dir}' in x or '{model}' in x    # noqa: E731
        targets = []
        for i, (raw, x) in enumerate(words):
            if raw in ('>', '>>', '1>', '1>>', '&>') and i + 1 < len(words):
                targets.append(words[i + 1][1])
            elif re.match(r'^(&|1)?>>?[^&]', raw):
                targets.append(re.sub(r'^(&|1)?>>?', '', x))
        cmd = [x for _, x in words]
        if cmd and Path(cmd[0]).name in self.WRITERS:
            name = Path(cmd[0]).name
            args = [a for a in cmd[1:] if not a.startswith('-') or a.startswith('of=')]
            if name == 'dd':
                targets += [a[3:] for a in cmd if a.startswith('of=')]
            elif name in ('sed', 'perl'):
                if any(a in ('-i', '--in-place') or re.match(r'^-[a-zA-Z]*i', a) for a in cmd[1:]):
                    targets += args[1:]
            elif name in ('python', 'python3'):
                targets += [a for a in args if staged(a)]
            else:
                targets += args[-1:] if name in ('cp', 'mv', 'install', 'rsync', 'ln') else args
        if any(staged(t) for t in targets):
            self.writes.append(f'{self.path}:{line}: writes a staged input: {st[:100]}')

    # -- statements whose effect is unknown --
    def assigned(self, st: str):
        """Names a statement assigns or exports (after any `[ ... ] &&` guards), and whether it exports them."""
        st = self.guarded(st)[1]
        m = re.match(r'(export|declare|local|readonly|typeset)\s+(.*)', st, re.S)
        if m:
            names = [w.split('=', 1)[0].rstrip('+') for w, _ in _words(m.group(2)) if not w.startswith('-')]
            return names, m.group(1) == 'export' or (m.group(1) == 'declare' and '-x' in m.group(2).split())
        m = re.match(r'read\s+(.*)', st)
        if m:
            return [w for w, _ in _words(m.group(1)) if re.fullmatch(r'[A-Za-z_]\w*', w)], False
        m = re.match(r'([A-Za-z_]\w*)\+?=', st)
        return ([m.group(1)] if m else []), False

    def taint_names(self, names, exported=False):
        for name in names:
            if name in PINNED:
                continue
            mark = '{?' + name + '}'
            if name in ARRAYS or name in self.arrays:
                self.arrays[name] = self.arrays.get(name, []) + [mark]
            self.vars[name] = mark
            if exported or name in self.exports:
                self.exports[name] = mark

    def unknown_statement(self, line, st, why):
        """A statement under a guard or block this interpreter cannot evaluate: what it assigns or exports becomes
        unknown; touching the arrays, exporting or sourcing is reported."""
        st = re.sub(r'^\(?[^\s()=]+\)\s*', '', st) if re.match(r'^\(?[^\s()=]+\)', st) else st   # a case pattern
        names, exported = self.assigned(st)
        self.taint_names(names, exported)
        body = self.guarded(st)[1]
        if self.touches(st) or exported or re.match(r'(source|\.)\s', body):
            self.unparsed.append(f'{self.path}:{line}: {why}: {st[:80]}')

    def source(self, line, st):
        """`source FILE` / `. FILE`: a repository file ($CODE/...) is followed in place; anything else is reported."""
        words = self.words(st.split(None, 1)[1], line) if len(st.split(None, 1)) > 1 else []
        target = words[0] if words else ''
        rel = target[len('{code}/'):] if target.startswith('{code}/') else None
        text = self.reader(rel) if (rel and self.reader) else None
        if text is None or self.depth >= 5:
            self.unparsed.append(f'{self.path}:{line}: sources a file it cannot read here: {target[:80]}')
            return
        old, self.path = self.path, rel
        self.depth += 1
        try:
            self._run(_split_statements(text))
        finally:
            self.path, self.depth = old, self.depth - 1

    def statement(self, line: int, st: str):
        conds, body = self.guarded(st)
        if conds:
            results = [self.test(c, line) for c in conds]
            if None in results:
                self.unknown_statement(line, st, 'guard it cannot evaluate')
                return
            if not all(results):
                return
        st = body
        if re.match(r'(source|\.)\s', st):
            self.source(line, st)
            return
        if st.startswith('export ') or st.startswith('declare -x '):
            for w, q in _words(st.split(None, 1)[1] if st.startswith('export ') else st[len('declare -x '):]):
                if w.startswith('-'):
                    continue
                k, eq, v = w.partition('=')
                if eq:
                    val = ' '.join(self.expand_word(v, True, line))
                    if k not in PINNED:
                        self.vars[k] = val
                    self.exports[k] = val
                else:
                    self.exports[k] = self.vars.get(k, '{?' + k + '}')
            return
        if st.startswith('while ') and 'SERVE_ENV+=' in st:
            m = re.search(r'<\s*<\(\s*cfg\s+"(.*?)"\s*\|', st, re.S)
            if not m:
                self.unparsed.append(f'{self.path}:{line}: {st[:80]}')
            else:
                env = json.loads(self.cfg(m.group(1)) or '{}')
                self.arrays.setdefault('SERVE_ENV', []).extend(f'{k}={v}' for k, v in env.items())
            return
        m_arr = re.fullmatch(r'([A-Za-z_]\w*)(\+?)=\((.*)\)', st, re.S)
        if m_arr:
            name, add, inner = m_arr.groups()
            self.arrays[name] = (self.arrays.get(name, []) if add else []) + self.words(inner, line)
            return
        m_var = re.fullmatch(r'([A-Za-z_]\w*)=(.*)', st, re.S)
        if m_var:
            name, value = m_var.groups()
            if name in PINNED:
                return
            words = _words(value)
            if len(words) <= 1:
                self.vars[name] = ' '.join(self.expand_word(words[0][0], True, line)) if words else ''
                if name in self.exports:              # an exported variable's new value reaches the environment
                    self.exports[name] = self.vars[name]
            elif self.touches(st):
                self.unparsed.append(f'{self.path}:{line}: {st[:80]}')
            return
        if self.touches(st) and not st.startswith(('echo ', 'setsid ', 'timeout ', 'trap ')):
            self.unparsed.append(f'{self.path}:{line}: {st[:80]}')


@contextlib.contextmanager
def _tb_agents(tree: Tree):
    """tb_agents.pi_context_limit of this tree in sys.modules (loaded standalone: stdlib only), so common.py's
    finish_notice builds the real frozen notice record; sys.modules and sys.path are restored afterwards."""
    mod = tree.module('tb_agents/pi_context_limit.py', 'tb_agents_pi_context_limit')
    saved_mods, saved_path = dict(sys.modules), list(sys.path)
    try:
        if mod is not None:
            pkg = type(sys)('tb_agents')
            pkg.__path__ = []
            pkg.pi_context_limit = mod
            sys.modules['tb_agents'], sys.modules['tb_agents.pi_context_limit'] = pkg, mod
        yield mod
    finally:
        for k in set(sys.modules) - set(saved_mods):
            del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path[:] = saved_path


def eval_specs(tree: Tree) -> dict:
    """{specs, sources, skipped, warnings} for eval_q35: one spec per family x serving variant x GPU set x harness
    (keys without a harness suffix are pi, the default; others end in /<harness>)."""
    out = {'specs': {}, 'sources': {}, 'skipped': [], 'warnings': []}
    common = tree.module(f'{EVAL}/common.py', 'eval_common')
    submit_src, job = tree.read_text(f'{EVAL}/submit.py'), tree.read_text(f'{EVAL}/serve_job.sh')
    if common is None or submit_src is None or job is None:
        raise ExtractError(f'{tree.label}: {EVAL}/common.py, submit.py and serve_job.sh are all needed')
    num = tree.module(f'{EVAL}/serve_numerics.py', 'eval_serve_numerics')
    combos = []
    for variant in sorted(getattr(common, 'QWEN35_SERVING', {})):
        combos += [(f'qwen3_5/{variant}/{g}', ['--family', 'qwen3_5', '--serving', variant, '--serve-gpu', g])
                   for g in ('h100', 'h100x2', 'b200')]
    for variant, prof in sorted(getattr(common, 'LIGHTNING_SERVING', {}).items()):
        combos += [(f'nemotron_h/{variant}/{g}', ['--family', 'nemotron_h', '--serving', variant, '--serve-gpu', g])
                   for g in sorted(prof.get('gpus', {}))]
    if 'k2_horizon' in getattr(common, 'VLLM_VENVS', {}):
        combos.append(('k2_horizon/h100', ['--family', 'k2_horizon', '--serve-gpu', 'h100']))
    for fam in sorted(getattr(common, 'GRUG_TEMPLATES', {})):
        combos += [(f'{fam}/{g}', ['--family', fam, '--serve-gpu', g]) for g in sorted(common.SNOWBALL_PROFILES)]
    harnesses = sorted(getattr(common, 'PROTOCOLS', {'pi': None}), key=lambda h: (h != 'pi', h))
    templates = {Path(p).name: p for p in tree.list(f'{EVAL}/templates')}
    writes = set()
    with _tb_agents(tree):
        for base, args0 in combos:
            for harness in harnesses:
                key = base if harness == 'pi' else f'{base}/{harness}'
                args = [*args0, '--harness', harness]
                gpu = args[args.index('--serve-gpu') + 1]
                family = args[args.index('--family') + 1]
                try:
                    config = eval_config(submit_src, common, ['init', '--name', '{name}', '--model', '{model}', *args],
                                         family)
                    cc = common.SNOWBALL_PROFILES[gpu]['cc'] if family in getattr(common, 'GRUG_TEMPLATES', {}) \
                        else GPU_CC[gpu]
                    sj = ServeJob(job, config, cc, reader=tree.read_text).run()
                    fields, warn = eval_fields(sj, config, common, num, family, tree, templates)
                except ExtractError as e:
                    out['skipped'].append(f'{key}: {e}')
                    continue
                out['specs'][key] = fields
                out['sources'][key] = {'config': f'{EVAL}/submit.py init {" ".join(args)}',
                                       'serve_job': f'{EVAL}/serve_job.sh', 'common': f'{EVAL}/common.py'}
                out['warnings'] += [f'{key}: {w}' for w in warn] + [f'{key}: unparsed {u}' for u in sj.unparsed]
                writes.update(sj.writes)
    out['warnings'] += sorted(writes)     # once per script (every spec runs the same statements)
    return out


def _launch_env(launch: list, exports: dict):
    """(environment vLLM sees beyond the job's own, interpreter, launcher prefix) of serve_job.sh's LAUNCH: the
    `env K=V` assignments on top of the script's exports (native `env` without -i passes them on; `apptainer exec
    --cleanenv` passes only APPTAINERENV_* / SINGULARITYENV_* exports, without the prefix)."""
    if 'env' not in launch:
        return dict(exports), None, launch
    i = launch.index('env')
    launcher, j, flags = launch[:i], i + 1, []
    while j < len(launch) and launch[j].startswith('-'):
        flags.append(launch[j])
        j += 1
    if '-i' in flags or '--ignore-environment' in flags:
        env = {}
    elif any(x.endswith('apptainer') or x.endswith('singularity') for x in launcher) and '--cleanenv' in launcher:
        env = {k.split('ENV_', 1)[1]: v for k, v in exports.items() if k.startswith(('APPTAINERENV_', 'SINGULARITYENV_'))}
    else:
        env = dict(exports)
    while j < len(launch) and re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', launch[j]):
        k, v = launch[j].split('=', 1)
        env[k] = v
        j += 1
    return env, (launch[j] if j < len(launch) else None), launcher


def eval_fields(sj: ServeJob, config, common, num, family, tree, templates):
    launch, vargs = sj.arrays.get('LAUNCH', []), sj.arrays.get('VARGS', [])
    if not vargs:
        raise ExtractError(f'{sj.path}: no VARGS array built (the launch changed shape)')
    env, interpreter, launcher = _launch_env(launch, sj.exports)
    fields, warn = argv_fields(vargs)
    proto = config.get('protocol', {})
    sif = next((x for x in launcher if x.endswith('.sif')), None)
    venv = config.get('vllm_venv') or None
    fields.update({
        'weights.model': f'per run (--model; family {family})', 'weights.revision': 'per run',
        'weights.checkpoint': 'per run (--model)',
        'weights.quantization': _quant_label(num, vargs),
        f'{HOST}vllm.build': venv_build(venv) if venv else None, 'vllm.sif': sif,
        'vllm.cuda_home': env.get('CUDA_HOME') or ('SIF' if sif else None),
        'vllm.entrypoint': [interpreter.replace(venv or '\0', '{venv}') if interpreter else None]
        + [a for a in vargs[:3] if not a.startswith('--') and a != '{model_dir}'],
        'output_cap.tokens': fields.get('output_cap.tokens', proto.get('max_output_tokens')),
        'output_cap.kind': (fields['output_cap.kind'] if fields['output_cap.kind'] != 'none on the server' else
                            'client-side only (protocol max_output_tokens; none on the server)'),
        'request.pi_thinking': proto.get('pi_thinking'),
        'request.overrides.temperature': proto.get('temperature'), 'request.overrides.top_p': proto.get('top_p'),
        'request.overrides.reasoning_effort': config.get('reasoning_effort') or (
            "proxy default (Pi's max)" if str(config.get('harness', 'pi')).startswith('pi') else 'harness default'),
        'protocol.harness': config.get('harness'),
        'protocol.agent': proto.get('agent'),
    })
    flatten('protocol.finish_notice', proto.get('finish_notice'), fields)
    for k in ('interleaved_thinking', 'enable_summarize', 'parser', 'context_policy'):
        if k in proto:
            fields[f'protocol.{k}'] = proto[k]
    tpl = fields.get('chat_template.path')
    if tpl == '{model_dir}/eval_chat_template.jinja':
        fields['chat_template.path'] = 'per run (--chat-template, default <model>/chat_template.jinja)'
    allowed = getattr(common, 'GRUG_TEMPLATES', {}).get(family)
    if allowed:
        fields['chat_template.allowed_sha256'] = sorted(allowed)
    for rel in [r for n, r in templates.items() if n.startswith(family)] + EVAL_FAMILY_TEMPLATES.get(family, []):
        fields[f'chat_template.repo.{rel}'] = tree.sha256(rel) or 'missing from the repository'
    fields.update(env_fields(env))
    add_repo_hashes(fields, tree, '{code}/')
    return fields, warn


# ---------------------------------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------------------------------
def trunk_pipelines(root) -> list[str]:
    """The numerics pipelines a checkout's package pipeline.json files name (the trunk, #480: one tree, every
    pipeline), in PIPELINES order."""
    r = subprocess.run(['git', 'ls-files', '--', '*pipeline.json'], cwd=root, capture_output=True, text=True)
    names = set()
    for rel in r.stdout.split():
        try:
            names.add(json.loads((Path(root) / rel).read_text()).get('pipeline'))
        except (OSError, ValueError, AttributeError):
            continue
    return [p for p in PIPELINES if p in names]


def extract_full(pipeline: str, root, ref: str | None = None) -> dict:
    """{specs, sources, skipped, warnings} of `pipeline` in `root` (a checkout; or, with `ref`, that ref of the
    repository at `root`)."""
    if pipeline not in PIPELINES:
        raise ExtractError(f'unknown pipeline {pipeline!r} (known: {", ".join(PIPELINES)})')
    tree = GitTree(root, ref) if ref else FsTree(root)
    out = eval_specs(tree) if pipeline == 'eval' else toml_specs(tree, pipeline)
    out['source'] = {'ref': ref, 'commit': getattr(tree, 'commit', None)}
    return out


def extract(pipeline: str, root, ref: str | None = None) -> dict:
    """{spec_key: {field: value}}: only NUMERICS (and `adjacent.*`) fields; keys are stable across checkouts."""
    return extract_full(pipeline, root, ref)['specs']


def load_record(path=RECORD) -> dict:
    """The record, with every section's specs decoded to full field sets."""
    p = Path(path)
    if not p.exists():
        return {'schema': SCHEMA, 'pipelines': {}}
    rec = json.loads(p.read_text())
    for sec in rec.get('pipelines', {}).values():
        sec['specs'] = decode_specs(sec.get('specs', {}))
    return rec


def _base_of(key: str, fields: dict, specs: dict):
    """The spec a stored spec is a delta against: a campaign's teacher spec, a tuning candidate's campaign."""
    if '/candidate/' in key:
        b = key.split('/candidate/')[0]
        return b if b in specs else None
    b = fields.get('campaign.teacher')
    return b if key.startswith('campaign/') and b in specs and not b.startswith('campaign/') else None


def encode_specs(specs: dict) -> dict:
    """Stored form: a campaign as {"$base": teacher, "$set": {...}, "$unset": [...]} against its teacher spec, a
    candidate against its campaign; every other spec in full."""
    out = {}
    for key, f in specs.items():
        b = _base_of(key, f, specs)
        if b is None:
            out[key] = f
            continue
        base = specs[b]
        out[key] = {'$base': b, '$set': {k: v for k, v in f.items() if k not in base or base[k] != v},
                    '$unset': sorted(set(base) - set(f))}
    return out


def decode_specs(stored: dict) -> dict:
    out = {}

    def get(key, depth=0):
        if key in out:
            return out[key]
        f = stored[key]
        if isinstance(f, dict) and '$base' in f:
            if depth > 4 or f['$base'] not in stored:
                raise ExtractError(f'record: spec {key} has an unknown or cyclic $base {f["$base"]!r}')
            full = {k: v for k, v in get(f['$base'], depth + 1).items() if k not in set(f.get('$unset', []))}
            full.update(f.get('$set', {}))
            f = full
        out[key] = f
        return f
    for key in stored:
        get(key)
    return {k: out[k] for k in stored}


def _jsonable(x):
    return json.loads(json.dumps(x, sort_keys=True))


def _unreadable(v) -> bool:
    return isinstance(v, str) and v.startswith(UNREADABLE) or (
        isinstance(v, list) and any(isinstance(x, str) and x.startswith(UNREADABLE) for x in v))


def compare(want: dict, have: dict, host_skipped: list | None = None) -> list[str]:
    """Every changed, added or removed spec / field between a record (`want`) and an extraction (`have`).

    Host-derived fields (`host.*`: checkpoint, venv, node templates and kernel dirs) are compared too, except where
    either side could not read its source (an UNREADABLE value): those fields, and every host field under the same
    prefix, are skipped and listed in `host_skipped` instead (a node without /data must never fail a pipeline)."""
    want, have = _jsonable(want), _jsonable(have)
    out = []
    for key in sorted(set(want) | set(have)):
        if key not in have:
            out.append(f'removed spec {key}')
            continue
        if key not in want:
            out.append(f'added spec {key}: ' + json.dumps(have[key], sort_keys=True)[:300])
            continue
        w, h = want[key], have[key]
        dark = sorted({f for side in (w, h) for f, v in side.items() if f.startswith(HOST) and _unreadable(v)})
        for f in sorted(set(w) | set(h)):
            if f.startswith(HOST) and any(f == d or f.startswith(d + '.') or d.startswith(f + '.') for d in dark):
                if host_skipped is not None and (w.get(f) != h.get(f) or _unreadable(h.get(f))):
                    host_skipped.append(f'{key}: {f}: {json.dumps(h.get(f, w.get(f)))[:120]}')
                continue
            tag = ' (numerics-adjacent)' if f.startswith('adjacent.') else ''
            if f not in h:
                out.append(f'{key}: removed {f}{tag} (was {json.dumps(w[f])})')
            elif f not in w:
                out.append(f'{key}: added {f}{tag} = {json.dumps(h[f])}')
            elif w[f] != h[f]:
                out.append(f'{key}: changed {f}{tag}: {json.dumps(w[f])} -> {json.dumps(h[f])}')
    return out


def _norm_warning(w: str) -> str:
    """A warning without its line numbers (`file:123:` -> `file:`), so unrelated edits that move lines do not count."""
    return re.sub(r'(\.(?:sh|py|toml)):\d+(?=[:\s])', r'\1', w)


def compare_warnings(want: list, have: list) -> list[str]:
    """Warnings (statements the extractor could not follow, specs it rendered with placeholders) that are new or
    gone: an unparsed change to the launch can change outputs, so it fails like a field change."""
    w, h = {_norm_warning(x) for x in want}, {_norm_warning(x) for x in have}
    return ([f'new extractor warning (it cannot follow this; it may change numerics): {x}' for x in sorted(h - w)]
            + [f'extractor warning gone: {x}' for x in sorted(w - h)])


def check_full(pipeline: str, root, record_path=RECORD, extraction: dict | None = None) -> tuple[list[str], list[str]]:
    """(failures, host fields skipped because their node source could not be read). `extraction`: a dict that gets
    the extracted specs (known_trunk_gaps compares added specs with them)."""
    rec = load_record(record_path)['pipelines'].get(pipeline)
    if rec is None:
        return [f'pipeline {pipeline} has no section in {Path(record_path).name}'], []
    full = extract_full(pipeline, root)
    if extraction is not None:
        extraction.update(full['specs'])
    host = []
    out = compare(rec['specs'], full['specs'], host)
    out += [f'skipped (could not extract): {s}' for s in full['skipped'] if s not in rec.get('skipped', [])]
    out += [f'no longer skipped: {s}' for s in rec.get('skipped', []) if s not in full['skipped']]
    out += compare_warnings(rec.get('warnings', []), full['warnings'])
    return out, host


def check(pipeline: str, root, record_path=RECORD) -> list[str]:
    return check_full(pipeline, root, record_path)[0]


def record(pipeline: str, root, ref: str | None = None, record_path=RECORD, force: bool = False) -> dict:
    """Write `pipeline`'s section of the record from `root` (or `ref`); keeps the other pipelines' sections and this
    section's notes for specs that still exist."""
    full = extract_full(pipeline, root, ref)
    dark = [f'{key}: {f}' for key, fields in full['specs'].items() for f, v in fields.items()
            if f.startswith(HOST) and _unreadable(v)]
    if dark:
        print(f'{pipeline}: {len(dark)} host-derived fields unreadable on this node' +
              (' (recorded anyway: --force)' if force else ''), file=sys.stderr)
        for d in dark[:10]:
            print(f'  {d}', file=sys.stderr)
        if not force:
            raise SystemExit(f'{pipeline}: refusing to record {len(dark)} unreadable host-derived fields (a record '
                             'made where /data or a venv cannot be read would never check them): record on a node '
                             'that reads them, or pass --force')
    p = Path(record_path)
    rec = json.loads(p.read_text()) if p.exists() else {'schema': SCHEMA, 'pipelines': {}}   # stored form kept
    old = rec['pipelines'].get(pipeline, {})
    notes = {k: v for k, v in (old.get('notes') or {}).items() if k in full['specs']}
    rec['schema'] = SCHEMA
    rec['note'] = ('Numerics fields of every serving spec per pipeline (tbshared/numerics.py; README "Numerics guard"). '
                   'specs, skipped and warnings (without line numbers) are compared by test_numerics.py; notes, '
                   'sources and recorded_from are informative. A spec with "$base" is stored as a delta against '
                   'that spec ($set / $unset).')
    rec['pipelines'][pipeline] = {
        'recorded_from': full['source'], 'specs': encode_specs(_jsonable(full['specs'])), 'notes': notes,
        'skipped': full['skipped'],
        'warnings': full['warnings'], 'sources': full['sources']}
    rec['pipelines'] = dict(sorted(rec['pipelines'].items()))
    p.write_text(json.dumps(_jsonable(rec), indent=1, sort_keys=False, ensure_ascii=False) + '\n')
    return {**rec['pipelines'][pipeline], 'specs': full['specs']}


# --- diff ------------------------------------------------------------------------------------------------------------
# Which model a spec serves, for the cross-pipeline table: the spec's weights.model, or for eval_q35 families the
# released model their checkpoints derive from (per-run checkpoints are SFT exports of it).
EVAL_FAMILY_MODELS = {'qwen3_5': 'Qwen/Qwen3.5-4B', 'nemotron_h': 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16',
                      'k2_horizon': 'k2-horizon-3.7b (dense base)', 'snowball': 'snowball GrugMoe 67B-A2B',
                      'datakit': 'snowball GrugMoe 67B-A2B'}


def spec_model(pipeline: str, key: str, fields: dict) -> str:
    if pipeline == 'eval':
        return EVAL_FAMILY_MODELS.get(key.split('/')[0], key.split('/')[0])
    return fields.get('weights.model') or key


def diff_rows(rec: dict, campaigns=False) -> list[tuple[str, str, dict]]:
    """[(model, field, {pipeline: {value: [spec keys]}})] for models in more than one pipeline, where the pipelines'
    value sets differ (a field one pipeline supplies per run, e.g. eval_q35's checkpoint, is left out)."""
    by_model = {}
    for p, sec in rec['pipelines'].items():
        for key, fields in sec['specs'].items():
            if key.startswith('campaign/') and not campaigns:   # serving specs only, as the inventory's section 2
                continue
            by_model.setdefault(spec_model(p, key, fields), {}).setdefault(p, {})[key] = fields
    rows = []
    for model in sorted(by_model):
        pipes = by_model[model]
        if len(pipes) < 2:
            continue
        fields = sorted({f for specs in pipes.values() for s in specs.values() for f in s})
        for f in fields:
            per = {p: {} for p in pipes}
            for p, specs in pipes.items():
                for key, s in specs.items():
                    per[p].setdefault(json.dumps(s.get(f, '<absent>'), sort_keys=True), []).append(key)
            per_run = any(all(v.startswith('"per run') for v in vals) for vals in per.values())
            if len({frozenset(v) for v in per.values()}) > 1 and not per_run:
                rows.append((model, f, per))
    return rows


def print_diff(rec: dict, file=sys.stdout, campaigns=False):
    last = None
    for model, f, per in diff_rows(rec, campaigns):
        if model != last:
            print(f'\n## {model}', file=file)
            last = model
        print(f'  {f}', file=file)
        for p, vals in per.items():
            for v, keys in vals.items():
                print(f'    {p:<15} {v[:90]:<90} {", ".join(sorted(keys))[:120]}', file=file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='op', required=True)
    for name in ('extract', 'record'):
        s = sub.add_parser(name)
        s.add_argument('pipeline', choices=PIPELINES)
        s.add_argument('root')
        s.add_argument('--ref', help='read the files of this git ref of the repository at ROOT')
        if name == 'record':
            s.add_argument('--force', action='store_true', help='record even if host-derived fields are unreadable')
    c = sub.add_parser('check')
    c.add_argument('pipeline', nargs='?')
    c.add_argument('root', nargs='?', default='.')
    dg = sub.add_parser('digest', help='TRUNK_PINNED digest of one spec')
    dg.add_argument('pipeline', choices=PIPELINES)
    dg.add_argument('spec')
    dg.add_argument('root', nargs='?', default='.')
    dp = sub.add_parser('diff')
    dp.add_argument('--campaigns', action='store_true', help='include synth campaign specs')
    a = ap.parse_args(argv)
    if a.op == 'extract':
        print(json.dumps(_jsonable(extract_full(a.pipeline, a.root, a.ref)), indent=1))
    elif a.op == 'record':
        sec = record(a.pipeline, a.root, a.ref, force=a.force)
        print(f'{a.pipeline}: {len(sec["specs"])} specs recorded from {sec["recorded_from"]}; '
              f'skipped {len(sec["skipped"])}, warnings {len(sec["warnings"])} -> {RECORD}')
        for s in sec['skipped'] + sec['warnings']:
            print(f'  {s}')
    elif a.op == 'digest':
        print(spec_digest(extract(a.pipeline, a.root)[a.spec]))
    elif a.op == 'check':
        r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=a.root, capture_output=True, text=True)
        root = r.stdout.strip() if r.returncode == 0 else a.root
        trunk = (Path(root) / 'tbshared' / 'TRUNK').is_file()
        if a.pipeline is not None:
            todo = [a.pipeline]
        elif trunk:   # the trunk: every pipeline a package names
            todo = trunk_pipelines(root)
        else:
            from tbshared import pipeline_config
            todo = [pipeline_config.pipeline_name(pipeline_config.PATH)]
        bad = 0
        for pipeline in todo:
            specs = {}
            d, host = check_full(pipeline, root, extraction=specs)
            for x in host:
                print(f'WARNING host source unreadable, not checked: {x}', file=sys.stderr)
            known = []
            if trunk:
                d, known = known_trunk_gaps(pipeline, d, specs=specs)
            if known:
                reasons = {}
                for _, why in known:
                    reasons[why] = reasons.get(why, 0) + 1
                print(f'{pipeline}: {len(known)} known trunk differences (TRUNK_KNOWN_GAPS):')
                for why, n in reasons.items():
                    print(f'  {n:4d}  {why}')
            print('\n'.join(d) or f'{pipeline}: numerics match the record' + (' apart from them' if known else ''))
            bad += bool(d)
        if bad:
            print(CHANGE_HINT)
        return 1 if bad else 0
    else:
        print_diff(load_record(), campaigns=a.campaigns)
    return 0


if __name__ == '__main__':
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
