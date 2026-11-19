"""The effective numerics of one vLLM engine as launched: a record for results, with a stable hash.

A spec's file says what a server should run; submit-time overrides (serve_teacher --tunables / --protocol, a
campaign's teacher_overrides, a tunable venv or extra_args / extra_env, per-replica arms) can change any of it. Each
serving job therefore writes what it actually started: the vLLM command line, the environment it set on top of the
job's own, the weights (checkpoint, revision, small config files' sha256, the weight quantization the engine applies),
the chat template's sha256, the vLLM build, the GPU model, and the client request defaults it publishes.

`sha256` covers every field that can change outputs, after the job-specific parts are abstracted (ports, the
node-local paths of the staged checkpoint and engine scratch, CUDA_VISIBLE_DEVICES): two engines with the same hash
ran the same recipe on the same GPU model. It is conservative: a throughput flag (max-num-seqs, memory utilization)
also changes it. Bookkeeping only: nothing here changes the command, the environment or the engine-cache key
(engine_cache.config_key).
"""
import hashlib
import json
from pathlib import Path

# Job-specific or numerics-free variables the launcher sets (GPU assignment, OpenMP threads, proxy bypass).
ENV_SKIP = ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'no_proxy', 'NO_PROXY', 'PYTHONNOUSERSITE',
            'FLASHINFER_NO_DOWNLOAD')   # a missing FlashInfer cubin fails instead of downloading: same artifacts
# Small checkpoint files that pin what the weights mean (architecture, quantization, stop ids, template).
CHECKPOINT_FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json', 'chat_template.jinja',
                    'chat_template.json', 'preprocessor_config.json')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def flag(argv, name):
    """The last value of `name` in argv (vLLM keeps the last), or None."""
    value = None
    for i, a in enumerate(argv[:-1]):
        if a == name:
            value = argv[i + 1]
    for a in argv:
        if a.startswith(name + '='):
            value = a.split('=', 1)[1]
    return value


def abstract(text, paths):
    for path, name in paths:
        if path:
            text = text.replace(str(path), name)
    return text


def portless(argv):
    """argv with every port value abstracted (as engine_cache.config_key does)."""
    args = list(argv)
    for i, a in enumerate(args[:-1]):
        if a == '--port' or (a.startswith('--') and a.endswith('-port')):
            args[i + 1] = '{port}'
    return args


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def weight_quantization(argv, model_dir):
    """What the engine does to the weights: the checkpoint's own quantization_config (or one injected by
    --hf-overrides), and vLLM's online --quantization on top. `label` is one short string for a results table."""
    config = read_json(Path(model_dir) / 'config.json') or {}
    overrides = flag(argv, '--hf-overrides')
    try:
        injected = (json.loads(overrides) or {}).get('quantization_config') if overrides else None
    except (ValueError, AttributeError):
        injected = None
    qc = injected or config.get('quantization_config') or (config.get('text_config') or {}).get('quantization_config')
    online = flag(argv, '--quantization') or None
    dtype = (config.get('torch_dtype') or config.get('dtype') or (config.get('text_config') or {}).get('torch_dtype')
             or (config.get('text_config') or {}).get('dtype'))
    checkpoint = None
    if qc:
        checkpoint = {k: qc.get(k) for k in ('quant_method', 'quant_algo', 'fmt', 'weight_block_size',
                                             'activation_scheme') if qc.get(k) is not None}
        checkpoint['source'] = 'hf-overrides' if injected else 'config.json'
    if checkpoint:
        label = '/'.join(str(x) for x in (checkpoint.get('quant_method'), checkpoint.get('quant_algo')
                                          or checkpoint.get('fmt')) if x)
        if online and online != checkpoint.get('quant_method'):
            label += f' + online {online}'
    elif online:
        label = f'{online} (online, from {dtype or "checkpoint dtype"})'
    else:
        label = str(dtype or 'checkpoint dtype')
    return {'label': label, 'online': online, 'checkpoint': checkpoint, 'checkpoint_dtype': dtype}


def template_identity(argv, model_dir, checkpoint=None):
    """The chat template the engine renders with: --chat-template's file, else the checkpoint's own."""
    explicit = flag(argv, '--chat-template')
    candidates = ([Path(explicit)] if explicit else
                  [Path(model_dir) / 'chat_template.jinja', Path(model_dir) / 'chat_template.json'])
    for path in candidates:
        if path.is_file():
            source = 'explicit' if explicit else 'checkpoint'
            shown = str(checkpoint) + '/' + path.name if checkpoint and not explicit else str(path)
            return {'source': source, 'path': shown, 'sha256': file_sha256(path)}
    tok = read_json(Path(model_dir) / 'tokenizer_config.json') or {}
    if tok.get('chat_template'):
        return {'source': 'tokenizer_config.json', 'path': None,
                'sha256': hashlib.sha256(json.dumps(tok['chat_template']).encode()).hexdigest()}
    return {'source': 'explicit (unreadable)' if explicit else 'none', 'path': explicit, 'sha256': None}


def vllm_build(venv):
    dists = sorted(p.name for p in Path(venv).glob('lib/python3*/site-packages/vllm-*.dist-info'))
    return dists[0][len('vllm-'):-len('.dist-info')] if dists else None


def replica_numerics(teacher, argv, env, *, model_dir, scratch_roots=(), protocol=None, venv=None, gpu=None,
                     base_env=None, overrides=None, declared_version=None):
    """The record (and its `sha256`) of one engine: see the module docstring. `scratch_roots` are the job's node-local
    roots (staging dir, engine scratch), abstracted so the hash is the same in every job; `base_env` is the
    environment the engine's env was built on (only what the launcher set or changed is recorded); `overrides` are
    the submit-time inputs (recorded, not hashed: their effect is in argv / env). `declared_version`: the version a
    tunable venv declares with it (tunables vllm_version), else the spec's runtime.vllm_version."""
    runtime = teacher.get('runtime', {})
    venv = venv or runtime.get('venv')
    roots = [(model_dir, '{model_dir}')] + [(r, '{scratch}') for r in sorted({str(r) for r in scratch_roots if r},
                                                                           key=len, reverse=True)]
    base = base_env or {}
    cache_env = set(runtime.get('cache_env', []))
    env_set = {k: abstract(str(v), roots) for k, v in sorted(env.items())
               if base.get(k) != v and k not in ENV_SKIP and k not in cache_env}
    files = {}
    for name in CHECKPOINT_FILES:
        path = Path(model_dir) / name
        if path.is_file():
            files[name] = file_sha256(path)
    template = template_identity(argv, model_dir, teacher.get('checkpoint'))
    template['path'] = abstract(template['path'], roots) if template['path'] else None
    numerics = {
        'argv': [abstract(a, roots) for a in portless(argv)],
        'env': env_set,
        'weights': {'model': teacher.get('model'), 'revision': teacher.get('revision'),
                    'checkpoint': teacher.get('checkpoint'), 'declared_quantization': teacher.get('quantization'),
                    'quantization': weight_quantization(argv, model_dir), 'files_sha256': files},
        'template': template,
        'vllm': {'venv': venv, 'version': vllm_build(venv) if venv else None,
                 'declared_version': declared_version or runtime.get('vllm_version')},
        'gpu': gpu,
        # only the protocol fields the engine command uses (vllm_launch: max_model_len, max_output_tokens): the
        # campaign protocol also carries client-side settings (agent timeouts, Pi version) that do not change numerics
        'protocol': {k: protocol.get(k) for k in ('context_tokens', 'max_output_tokens')},
        'request': teacher.get('request', {}),
    }
    return {**numerics, 'served_name': teacher.get('served_name'), 'teacher': teacher.get('slug'),
            **({'overrides': overrides} if overrides is not None else {}), 'sha256': digest(numerics)}


def job_numerics_sha256(replicas):
    """One hash for a job: its replicas' distinct hashes (twin engines of one recipe hash like a single engine)."""
    hashes = sorted({r.get('sha256') for r in replicas if r.get('sha256')})
    return hashes[0] if len(hashes) == 1 else (digest(hashes) if hashes else None)


def safe_replica_numerics(*args, **kwargs):
    """replica_numerics that never raises: bookkeeping must never stop a server."""
    try:
        return replica_numerics(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {'error': repr(exc)[:500], 'sha256': None}
