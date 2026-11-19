"""Pinned Pi continuation patch and loss annotations for its native session history."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile
import time

LEGACY_CACHE = Path('/data/aai/tbench/agent_cache_pi_cont')
LEGACY_SHA256 = 'e6188bb6e5f269106f00944174de1baeff10485ed6f85b7e0ea02d7f5b7fb336'
CACHE = Path('/data/aai/tbench/agent_cache_pi_cont_reasoning_v1')
STOCK_SHA256 = '315bf795edc8e6653180e43156c9d0ba60b4602fef55c2a7c077fa33168f9493'
VERSION = '1.2.0'
BUNDLE = Path('pi/1.2.0/linux-x64/harbor-pi.tar')
CONTINUE = 'Thinking exceeded the maximum thinking budget, please continue.'
OLD = 'explicitContinuation=decision?.action==="continue",pendingMessages=await config.getSteeringMessages?.()||[],(hasMoreToolCalls||pendingMessages.length>0)&&(explicitContinuation=!1)}'
INSERT = 'message.stopReason==="length"&&toolCalls.length===0&&pendingMessages.push({role:"user",content:[{type:"text",text:"' + CONTINUE + '"}],timestamp:Date.now()}),'
NEW = OLD.replace('||[],', '||[],' + INSERT)
LOOP_MEMBER = '/dist/bundle/chunks/chunk-TJXZSVTD.js'
REASONING_MEMBER = '/dist/bundle/chunks/openai-completions-QW2B4SML.js'
REASONING_OLD = 'let content=assistantMsg.content;if(!(content!=null&&content.length>0)&&!assistantMsg.tool_calls)continue;params.push(assistantMsg)'
# Only retain reasoning the provider converter already knows how to serialize.
# Empty required reasoning_content placeholders must not create empty assistants.
REASONING_NEW = ('let content=assistantMsg.content;'
    'if(!(content!=null&&content.length>0)&&!assistantMsg.tool_calls'
    '&&!OPENAI_COMPLETIONS_REASONING_FIELDS.some(field=>typeof assistantMsg[field]==="string"&&assistantMsg[field].trim().length>0)'
    '&&!(assistantMsg.reasoning_details?.length>0))continue;params.push(assistantMsg)')
PATCHES = ((LOOP_MEMBER, OLD, NEW), (REASONING_MEMBER, REASONING_OLD, REASONING_NEW))
# Bash default timeout (2026-10-21). Stock Pi bash has no default timeout, and commands that never
# return (interactive prompts, `server &` + `wait`, open nc sessions) held 1-4% of attempts for up
# to the 6 h agent limit and about half of all attempt-hours. Completed calls: p99 153 s, p99.9
# 639 s. The default and how to raise it are stated in the tool description the model sees.
BASH_CACHE = Path('/data/aai/tbench/agent_cache_pi_cont_bash180_v1')
BASH_DEFAULT_TIMEOUT_SEC = 180
BASH_PATCHES = (
    (LOOP_MEMBER, 'async execute(_toolCallId,{command,timeout},signal,onUpdate,ctx){',
     'async execute(_toolCallId,{command,timeout=180},signal,onUpdate,ctx){'),
    (LOOP_MEMBER, 'description:"Timeout in seconds (optional, no default timeout)"',
     'description:"Timeout in seconds (default: 180). Set a larger value for commands that need longer, '
     'such as builds, installs or test suites."'),
    (LOOP_MEMBER, 'If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.`',
     'If truncated, full output is saved to a temp file. Commands are killed after 180 seconds unless you '
     'provide a larger timeout in seconds.`'),
    (LOOP_MEMBER, 'throw new Error(appendStatus(text,`Command timed out after ${timeoutSecs} seconds`))',
     'throw new Error(appendStatus(text,`Command timed out after ${timeoutSecs} seconds and was killed. '
     'If it needs more time, run it again with a larger timeout. If it was waiting for input or for a '
     'process that never exits, run it non-interactively, or start the process in the background with '
     'its output redirected to a file.`))'),
)
PATCH_SETS = {'reasoning_v1': PATCHES, 'bash180_v1': PATCHES + BASH_PATCHES}


def launch_cache(env=None):
    """Use the fixed bundle for new attempts, including workers with old overrides.

    Historical bundles remain verifiable explicitly for audits. The campaign owner
    authorized upgrading ongoing workers at their next invocation (issue #1).
    """
    env = os.environ if env is None else env
    cache = Path(env.get('PI_CONT16K_CACHE', str(CACHE)))
    return CACHE if cache.resolve() == LEGACY_CACHE.resolve() else cache


def patch_source(source, old, new):
    """Fail closed on a different upstream version or an already patched input."""
    if source.count(old) != 1 or new in source:
        raise ValueError('Expected exactly one stock Pi patch location')
    return source.replace(old, new)


def resampling_policy(env=None):
    """Default to no resampling; honor immutable evaluation manifests.

    Older running workers predate PI_CONT16K_RUN, so recover their run directory
    from Condor's local job ad. This also works for replica queues whose results
    link to the central manifest, without restarting active trials.
    """
    env = os.environ if env is None else env
    run = env.get('PI_CONT16K_RUN')
    if not run and env.get('_CONDOR_JOB_AD'):
        for line in Path(env['_CONDOR_JOB_AD']).read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip() in ('Args', 'Arguments'):
                args = shlex.split(json.loads(value.strip()))
                if args and Path(args[0]).is_absolute():
                    run = args[0]
                break
    explicit = env.get('PI_CONT16K_RESAMPLES')
    count, source = (int(explicit), 'environment') if explicit is not None else (0, 'default')
    if run:
        root = Path(run)
        for manifest in (root / 'manifest.json', (root / 'results').resolve().parent / 'manifest.json'):
            if not manifest.is_file():
                continue
            policy = json.loads(manifest.read_text()).get('pi_continuation', {})
            if 'proxy_resamples' in policy:
                pinned = policy['proxy_resamples']
                if type(pinned) is not int or pinned < 0:
                    raise ValueError(f'Invalid resampling policy in {manifest}')
                if explicit is not None and count != pinned:
                    raise ValueError(f'PI_CONT16K_RESAMPLES conflicts with the evaluation manifest: {manifest}')
                count, source = pinned, str(manifest)
                break
    if count < 0:
        raise ValueError('PI_CONT16K_RESAMPLES must be nonnegative')
    return count, source


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def verify_bundle(cache=CACHE):
    path = Path(cache) / BUNDLE
    expected = path.with_suffix('.tar.sha256').read_text().split()[0]
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Pi continuation bundle checksum mismatch: {path}')
    legacy = Path(cache).resolve() == LEGACY_CACHE.resolve() and actual == LEGACY_SHA256
    sources = {}
    with tarfile.open(path) as tf:
        for suffix in {suffix for suffix, _, _ in PATCHES + BASH_PATCHES}:
            names = [n for n in tf.getnames() if n.endswith(suffix)]
            if len(names) != 1:
                raise ValueError(f'Cannot identify the pinned Pi 1.2.0 chunk: {suffix}')
            sources[suffix] = tf.extractfile(names[0]).read().decode()
    for suffix, old, new in PATCHES:
        source = sources[suffix]
        if legacy and suffix == REASONING_MEMBER:
            if source.count(old) != 1 or new in source:
                raise ValueError('Historical Pi reasoning adapter changed')
        elif source.count(new) != 1 or old in source:
            raise ValueError(f'Pi patch missing or ambiguous: {suffix}')
    # The bash default timeout is all or nothing.
    applied = [sources[s].count(new) == 1 and old not in sources[s] for s, old, new in BASH_PATCHES]
    stock = [sources[s].count(old) == 1 and new not in sources[s] for s, old, new in BASH_PATCHES]
    if not (all(applied) or all(stock)):
        raise ValueError('Pi bash default-timeout patch partially applied or ambiguous')
    _VERIFIED[actual] = BASH_DEFAULT_TIMEOUT_SEC if all(applied) else None
    return actual


_VERIFIED = {}  # bundle sha256 -> bash default timeout (None: stock, no default)


def bundle_provenance(cache, sha256):
    return {'bundle_cache': str(cache), 'bundle_sha256': sha256,
            'reasoning_history_patch': not (Path(cache).resolve() == LEGACY_CACHE.resolve()
                                           and sha256 == LEGACY_SHA256),
            'bash_default_timeout_sec': _VERIFIED.get(sha256)}


def session_annotations(logs_dir):
    """Build per-message SFT loss weights without changing native events.

    This deliberately preserves branches, compaction events, tool calls and tool results.
    Consumers must reconstruct Pi's session context, not flatten the event log.
    """
    from tb_agents.response_budget import CAP_MARK as RESPONSE_CAP_MARK
    sessions = []
    errors = []
    for path in sorted(Path(logs_dir).glob('pi/sessions/**/*.jsonl')):
        entries = []
        # Native JSONL uses LF record boundaries. Unicode NEL/line/paragraph
        # separators can occur unescaped inside JSON strings (tool observations,
        # reasoning) and must not split records (str.splitlines would). Exact bytes, no newline translation (#326).
        for lineno, line in enumerate(path.read_bytes().decode().split('\n'), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                errors.append({'source': str(path.relative_to(logs_dir)), 'line': lineno})
                continue
            message = entry.get('message') or {}
            if entry.get('type') == 'message':
                entry['sft_loss_weight'] = int(message.get('role') == 'assistant'
                    and message.get('stopReason') in ('stop', 'toolUse')
                    and not any(b.get('type') == 'text' and b.get('text') in (
                        '[harbor-gen context cap reached; episode ended by the generation harness]', RESPONSE_CAP_MARK)
                        for b in message.get('content', []) if isinstance(b, dict)))
            entries.append(entry)
        sessions.append({'source': str(path.relative_to(logs_dir)),
                         'sha256': digest(path), 'entries': entries})
    return {'schema': 'pi-native-sft-events-v1', 'sessions': sessions,
            'parse_errors': errors, 'valid_for_export': bool(sessions) and not errors,
            'continuation_message': CONTINUE,
            'loss_rule': 'Only complete assistant messages receive loss; length/error/aborted messages remain context with zero loss.'}


def annotate_sessions(logs_dir):
    """Publish complete annotations atomically, keeping any previous file on failure.

    Lustre can fail a write after truncating its destination, so the sidecar is
    written to a unique temporary file, fsynced and renamed. Only transient
    filesystem errors (EIO, ESTALE) are retried; an exhausted or other failure
    is raised and never leaves a partial sidecar or temporary file behind.
    """
    output = session_annotations(logs_dir)
    target = Path(logs_dir) / 'pi/sft-events.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(output, ensure_ascii=False) + '\n'
    for attempt in range(3):
        temporary = target.with_name(f'{target.name}.{os.getpid()}.{attempt}.tmp')
        try:
            temporary.write_text(data, encoding='utf-8')
            fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, target)
            return output
        except OSError as exc:
            if exc.errno not in (errno.EIO, errno.ESTALE) or attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)
