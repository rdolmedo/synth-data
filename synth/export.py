"""Pi graph export and unique-source-token accounting; never sum API prompt usage.

  python -m synth.export --kind train|eval|smoke|all     (SYNTH_CAMPAIGN=...)

Native event IDs are counted once. Branch/compaction boundaries carry context
references, not copied prefixes. The system prompt and tool schemas come from
the first served request in the attempt's wire log and are counted once per
trajectory. Loss weight is 1 only for complete assistant turns (stop/toolUse)
that no shared annotation zeroes; the context guard's synthetic stop is 0.
Non-assistant and truncated turns still count as context tokens.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

from . import archive
from .common import REPO, atomic, harbor_result, pi_log_paths, read, sha
from .config import Campaign
from .recover_native import recover_sparse_terminal, recover_tail, recover_wire_gap
from .serializers import serializer_for
from .wire import load as load_wire, sse_events

sys.path.insert(0, str(REPO))
from tb_agents.pi_context_limit import CAP_MARK  # noqa: E402

EXPORTER_VERSION = 'synth-pi-source-graph-v1'
OMITTED_IMAGE_NOTE = '[Current model does not support images. The image will be omitted from this request.]'


def text_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    if not isinstance(content, list):
        raise ValueError('Malformed native message content')
    # Pi keeps image bytes in its native session even when the served model is
    # text-only; its tool result records the omission. Keep the bytes, count none.
    text = ''.join(p.get('text', '') for p in content if p.get('type') == 'text')
    allowed = ('text', 'thinking', 'toolCall', 'image') if OMITTED_IMAGE_NOTE in text else ('text', 'thinking', 'toolCall')
    if any(p.get('type') not in allowed for p in content):
        raise ValueError('Unsupported nontext content; preserve raw data for a future multimodal exporter')
    return text


def convert(message):
    role = message['role']
    parts = message.get('content') if isinstance(message.get('content'), list) else []
    attachments = [p for p in parts if p.get('type') == 'image']
    omitted = {'native_omitted_attachments': attachments,
        'attachment_accounting': 'Pi explicitly omitted these images from the served text request; bytes retained, no vision tokens counted.'} if attachments else {}
    if role == 'system':
        content = message.get('content') or '\n\n'.join(message.get('sections', {}).values())
        return {'role': 'system', 'content': content}
    if role == 'toolResult':
        return {'role': 'tool', 'tool_call_id': message['toolCallId'], 'name': message.get('toolName'),
                'content': text_content(message.get('content')), **omitted}
    result = {'role': role, 'content': text_content(message.get('content')), **omitted}
    if role == 'assistant':
        result['reasoning_content'] = ''.join(p.get('thinking', '') for p in parts if p.get('type') == 'thinking')
        calls = [p for p in parts if p.get('type') == 'toolCall']
        if calls:
            result['tool_calls'] = [{'id': p['id'], 'type': 'function',
                'function': {'name': p['name'], 'arguments': p.get('arguments', {})}} for p in calls]
        if message.get('stopReason'):
            result['_stop_reason'] = message['stopReason']
    return result


def wire_logs(folder):
    """The proxy writes under .pi-cont16k/<name>-*/ and moves it into the trial at teardown."""
    return sorted({*folder.glob('harbor/pi-cont16k/wire.jsonl'), *folder.glob('.pi-cont16k/*/wire.jsonl')})


def signature(folder):
    files = (pi_log_paths(folder, 'sessions/*.jsonl') + sorted(folder.glob('worker-result.json'))
             + pi_log_paths(folder, 'sft-events.json') + sorted(folder.glob('native-graph-recovery.json'))
             + wire_logs(folder))
    recovery = read(folder / 'native-graph-recovery.json')
    if recovery:
        files.append(folder / recovery['pi_output'])
    return [[str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def export_one(campaign, folder, serializer):
    item = read(folder / 'attempt.json')
    final = (folder / 'worker-result.json').exists()
    logs = wire_logs(folder)
    if len(logs) > 1:
        raise ValueError(f'Multiple wire logs for one attempt: {logs}')
    calls = load_wire(logs[0]) if logs else []
    wire = [c['body'] for c in calls if isinstance(c.get('body'), dict) and c['body'].get('messages')]
    native = pi_log_paths(folder, 'sessions/*.jsonl')
    if not native or not wire:
        raise ValueError('No native Pi session or served request with tools yet')
    first = next((r for r in wire if r.get('tools')), None)
    if first is None:
        raise ValueError('Missing actual Pi tool schemas')
    tools = first['tools']
    system = next((m for m in first['messages'] if m['role'] == 'system'), None)
    if system is None:
        raise ValueError('Missing actual system prompt')
    if any(r.get('tools', tools) != tools for r in wire):
        raise ValueError('Tool schema changed within a trajectory; requires separate schema segments')
    sessions, segments, seen = [], [], set()
    message_count = assistant_count = 0
    for path in native:
        entries, parse_errors = [], []
        # JSONL is LF-delimited; U+2028/NEL inside JSON strings are legal and
        # splitlines() would corrupt such records.
        with path.open(encoding='utf-8') as source:
            lines = list(source)
        for lineno, line in enumerate(lines, 1):
            try:
                entry = json.loads(line)
            except ValueError:
                if not final and lineno == len(lines) and not line.endswith('\n'):
                    continue
                parse_errors.append(lineno)
                continue
            entries.append(entry)
        parse_recovery = None
        if parse_errors:
            entries, parse_recovery = recover_sparse_terminal(folder, path, lines, parse_errors, entries)
        entries, graph_recovery = recover_tail(folder, path, entries)
        if parse_recovery:
            if graph_recovery is None:
                raise ValueError('Sparse terminal requires verified structured stdout recovery')
            graph_recovery['native_parse_recovery'] = parse_recovery
        entries, wire_recovery = recover_wire_gap(folder, path, entries, convert, calls)
        if wire_recovery:
            if graph_recovery:
                raise ValueError('Multiple native recovery rules for one session')
            graph_recovery = wire_recovery
        annotations = read(path.parent.parent / 'sft-events.json', {})
        shared_weights = {}
        for annotated in annotations.get('sessions', []):
            if str(path).endswith(annotated['source']) and annotated.get('sha256') == sha(path):
                shared_weights.update({e['id']: e['sft_loss_weight'] for e in annotated['entries']
                                       if e.get('id') and 'sft_loss_weight' in e})
        if not campaign.protocol['compaction'] and any(e.get('type') in ('compaction', 'branch_summary') for e in entries):
            atomic(campaign.ops / 'protocol-defect.json', {'time': time.time(), 'source': str(path),
                'reason': 'Automatic compaction/summary in a no-compaction campaign; admission paused.'})
            raise ValueError('Protocol defect: compaction is forbidden in this campaign')
        session_id = next((e['id'] for e in entries if e.get('type') == 'session'), str(path))
        by_id = {e['id']: e for e in entries if e.get('id') and e.get('type') != 'session'}
        if len(by_id) != sum(bool(e.get('id')) and e.get('type') != 'session' for e in entries):
            raise ValueError('Duplicate native event IDs')
        for e in entries:
            if e.get('parentId') and e['parentId'] not in by_id:
                raise ValueError(f'Missing parent {e["parentId"]}')
        segment = previous = None
        for e in entries:
            eid = e.get('id')
            if e.get('type') == 'session' or not eid:
                continue
            key = f'{session_id}/{eid}'
            if key in seen:
                raise ValueError(f'Duplicate event across sessions: {key}')
            seen.add(key)
            boundary = e.get('parentId') != previous or e.get('type') in ('compaction', 'branch_summary')
            if segment is None or boundary:
                ancestry, parent, visited = [], e.get('parentId'), set()
                while parent:
                    if parent in visited:
                        raise ValueError('Cycle in Pi parent graph')
                    visited.add(parent)
                    ancestry.append(parent)
                    parent = by_id[parent].get('parentId')
                ordered = list(reversed(ancestry))
                last_compaction = next((by_id[x] for x in reversed(ordered) if by_id[x].get('type') == 'compaction'), None)
                effective = ordered
                if last_compaction:
                    first_kept = last_compaction.get('firstKeptEntryId')
                    if first_kept and first_kept not in by_id:
                        raise ValueError('Compaction firstKeptEntryId missing from native graph')
                    kept_at = ordered.index(first_kept) if first_kept in ordered else len(ordered)
                    compact_at = ordered.index(last_compaction['id'])
                    retained = [x for x in ordered[kept_at:compact_at] if by_id[x].get('type') == 'message']
                    effective = [last_compaction['id']] + retained + ordered[compact_at + 1:]
                segment = {'session_id': session_id, 'index': len(segments), 'parent_event_id': e.get('parentId'),
                    'context_event_ids': ordered, 'effective_context_event_ids': effective,
                    'last_compaction': last_compaction,
                    'context_reconstruction': 'Root-to-parent native path; after the last compaction use its summary, retained messages from firstKeptEntryId, and post-compaction messages. Exact served snapshots remain in the wire log.',
                    'boundary': e.get('type') if boundary else 'start',
                    'event_ids': [], 'messages': [], 'loss_weights': [], 'compactions': []}
                segments.append(segment)
            segment['event_ids'].append(eid)
            if e.get('type') == 'message':
                msg = e['message']
                weight = int(msg['role'] == 'assistant' and msg.get('stopReason') in ('stop', 'toolUse'))
                weight = min(weight, shared_weights.get(eid, 1), e.get('sft_loss_weight', 1), msg.get('sft_loss_weight', 1))
                if CAP_MARK in text_content(msg.get('content')):
                    weight = 0
                e['sft_loss_weight'] = weight
                if msg['role'] != 'system':
                    segment['messages'].append({'event_id': eid, 'message': convert(msg),
                                                'loss_weight': weight, 'stop_reason': msg.get('stopReason')})
                    message_count += 1
                    assistant_count += msg['role'] == 'assistant'
            elif e.get('type') in ('compaction', 'branch_summary'):
                summary = e.get('summary')
                segment['compactions'].append(e)
                if summary:
                    actual = next((m for r in wire for m in r.get('messages', [])
                                   if m.get('role') == 'user' and summary in text_content(m.get('content'))), None)
                    segment['messages'].append({'event_id': eid, 'message': actual or {'role': 'user', 'content': summary},
                        'loss_weight': 0, 'source': 'native_compaction_summary', 'exact_wire_wrapper': actual is not None})
            previous = eid
        sessions.append({'session_id': session_id, 'source': str(path.relative_to(folder)), 'sha256': sha(path),
                         'entries': entries, **({'graph_recovery': graph_recovery} if graph_recovery else {})})
    aborted = []
    worker_result = read(folder / 'worker-result.json', {})
    last = calls[-1] if calls else None
    if final and worker_result.get('interrupted') and last and not last['complete'] and last['response']:
        # The proxy flushes streamed bytes before Pi has the whole response; keep
        # that last unfinished response even if Pi died before writing it.
        events = sse_events(last['response'])
        response_id = next((e.get('id') for e in events if e.get('id')), None)
        native_responses = {e.get('message', {}).get('responseId') for s in sessions for e in s['entries']}
        if response_id and response_id not in native_responses:
            content = reasoning = ''
            tool_deltas = []
            for event in events:
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    content += delta.get('content') or ''
                    reasoning += delta.get('reasoning_content') or delta.get('reasoning') or ''
                    tool_deltas.extend(delta.get('tool_calls') or [])
            if content or reasoning:
                orphan = {'event_id': f'aborted-wire/{response_id}', 'stop_reason': 'aborted', 'loss_weight': 0,
                    'message': {'role': 'assistant', 'content': content, 'reasoning_content': reasoning,
                                '_stop_reason': 'aborted'},
                    'wire_call': last['id'], 'unfinished_tool_deltas': tool_deltas, 'unfinished_tool_tokens_counted': False}
                aborted.append(orphan)
                segments.append({'index': len(segments), 'boundary': 'aborted_backend_stream', 'messages': [orphan],
                                 'event_ids': [orphan['event_id']], 'context_wire_call': last['id'],
                                 'context_counted_again': False})
                assistant_count += 1
                message_count += 1
    if assistant_count < 1:
        raise ValueError('No saved assistant output yet')
    # Each disjoint segment is serialized once. System and schemas are counted
    # once per trajectory; later segment headers are subtracted.
    header = serializer.render([system], tools)
    total, first_segment = 0, True
    for segment in segments:
        if not segment['messages']:
            segment['tokens'] = 0
            continue
        rendered = serializer.render([system] + [x['message'] for x in segment['messages']], tools)
        if not rendered.startswith(header):
            raise ValueError('Serialization header is not a stable prefix')
        counted = rendered if first_segment else rendered[len(header):]
        segment['tokens'] = serializer.count(counted)
        segment['repeated_system_and_schema_tokens_excluded'] = 0 if first_segment else serializer.count(header)
        segment['serialized_sha256'] = hashlib.sha256(counted.encode()).hexdigest()
        total += segment['tokens']
        first_segment = False
    result = harbor_result(folder)
    teacher = campaign.teacher
    return {'schema': EXPORTER_VERSION, 'id': item['id'], 'attempt': item, 'finalized': final,
        'valid_for_training': item['kind'] == 'train', 'usable': True, 'tokens': total,
        'message_count': message_count, 'assistant_message_count': assistant_count,
        'system_message': system, 'tools': tools, 'sessions': sessions, 'segments': segments,
        'aborted_wire_messages': aborted, 'result': result, 'worker_result': worker_result,
        'tokenizer': {'repository': teacher['model'], 'revision': teacher['revision'], **serializer.identity()},
        'accounting': {'name': 'unique native event serialization with graph context references',
            'history_reasoning_retained': True, 'raw_request_snapshots_counted': False,
            'includes_context_tokens': True, 'training_chunks': None,
            'repeated_context': 'Referenced by parent/context IDs; excluded from the unique source-token count. A downstream chunker must report repeats separately.'},
        'wire_log': {'path': str(logs[0].relative_to(folder)), 'sha256': sha(logs[0]), 'calls': len(calls)},
        'updated_at': time.time()}


def reusable(entry):
    """A finished attempt's folder no longer changes: the worker writes worker-result.json after Harbor exits (the
    controller writes one only for a lost worker's attempts, once Condor confirms the job ended). Its usable export
    is reused without the signature check, which costs ~88 ms of Lustre globs and stats per folder (51 min per pass at
    35k trajectories even when nothing changed). Unusable exports are retried every pass, as before."""
    return bool(entry and entry.get('finalized') and entry.get('usable')
                and entry.get('exporter_version') == EXPORTER_VERSION)


def export_folder(campaign, serializer, kind, live, folder, old):
    """One attempt folder's index entry, or None while it is not exportable."""
    if not (folder / 'attempt.json').exists():
        return None
    if not live and kind != 'smoke' and not (folder / 'worker-result.json').exists():
        return None
    sig = signature(folder)
    if old and old.get('usable') and old.get('signature') == sig and old.get('exporter_version') == EXPORTER_VERSION:
        return old
    root = campaign.root_for(kind)
    try:
        record = export_one(campaign, folder, serializer)
        dst = root / 'exports' / f'{folder.name}.json.gz'
        dst.parent.mkdir(parents=True, exist_ok=True)
        # One write per file, hashed in memory: re-reading a small file on Lustre costs as much as writing it.
        data = gzip.compress(json.dumps(record, ensure_ascii=False).encode())
        tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.tmp')
        tmp.write_bytes(data)
        tmp.replace(dst)
        return {'id': folder.name, 'usable': True, 'tokens': record['tokens'],
            'exporter_version': EXPORTER_VERSION, 'finalized': record['finalized'], 'signature': sig,
            'path': str(dst), 'sha256': hashlib.sha256(data).hexdigest(), 'task': record['attempt']['task'],
            'attempt': record['attempt']['attempt'], 'retry': record['attempt'].get('retry', 0), 'kind': kind,
            'raw': str(folder), 'native_sessions': [s['source'] for s in record['sessions']]}
    except Exception as exc:
        entry = {'id': folder.name, 'usable': False, 'tokens': 0, 'exporter_version': EXPORTER_VERSION,
            'finalized': (folder / 'worker-result.json').exists(), 'signature': sig, 'error': repr(exc),
            'raw': str(folder), 'task': (read(folder / 'attempt.json') or {}).get('task')}
        if entry['finalized']:
            atomic(root / 'quarantine' / f'{folder.name}.json', entry)
        return entry


_worker = None


def _init_worker(config):
    global _worker
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.nice(10)  # the controller's 15 s tick shares these CPUs and comes first
    campaign = Campaign(config)
    _worker = (campaign, serializer_for(campaign.teacher))


def _pool_export(kind, live, folder, old):
    return export_folder(*_worker, kind, live, Path(folder), old)


class Stopped(BaseException):
    """SIGTERM (a controller restart or close): checkpoint the index and exit."""


def run_once(campaign, kind, serializer, workers=1, checkpoint_sec=300):
    """One pass over the attempt folders; new and changed ones are exported by `workers` processes (spawned only when
    there is work). The index is replaced as a whole, never edited. It is also checkpointed every `checkpoint_sec`
    and when the pass stops early, so a controller restart resumes the pass instead of discarding it."""
    root = campaign.root_for(kind)
    path = root / 'export-index.json'
    previous = read(path, {})
    # Live (in-flight) exports only matter when the stop rule counts live tokens.
    live = campaign.config.get('export_live', (campaign.config.get('train') or {}).get('mode', 'tokens') == 'tokens')
    folders = sorted((root / 'raw').glob('*'))
    # Attempts packed into shards (synth/archive.py) have no folder any more; their final entries are kept.
    names = {f.name for f in folders}
    packed = {name: previous[name] for name in archive.load_index(campaign.root, kind)['attempts']
              if name in previous and name not in names}
    index, todo = {}, []
    for folder in folders:
        if reusable(previous.get(folder.name)):
            index[folder.name] = previous[folder.name]
        else:
            todo.append(folder)
    pending = {f.name for f in todo if f.name in previous}
    last = time.time()

    def done(folder, entry):
        nonlocal last
        pending.discard(folder.name)
        if entry is not None:
            index[folder.name] = entry
        if time.time() - last >= checkpoint_sec:
            checkpoint()
            last = time.time()

    def checkpoint():
        atomic(path, {**packed, **{name: previous[name] for name in pending}, **index})

    pool = None
    try:
        if workers > 1 and len(todo) > workers:
            pool = concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                                          initializer=_init_worker, initargs=(campaign.config,))
            queue, inflight = iter(todo), {}
            while True:
                while len(inflight) < 4 * workers and (folder := next(queue, None)) is not None:
                    inflight[pool.submit(_pool_export, kind, live, str(folder), previous.get(folder.name))] = folder
                if not inflight:
                    break
                finished, _ = concurrent.futures.wait(inflight, timeout=10,
                                                      return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    done(inflight.pop(future), future.result())
        else:
            for folder in todo:
                done(folder, export_folder(campaign, serializer, kind, live, folder, previous.get(folder.name)))
    except BaseException:
        checkpoint()
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    index = {**packed, **{f.name: index[f.name] for f in folders if f.name in index}}
    atomic(path, index)
    return index


def stop(signum, frame):
    raise Stopped()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['train', 'eval', 'smoke', 'all'], default='all')
    ap.add_argument('--workers', type=int, help='export processes (default: export_workers, else the CPUs this job has)')
    a = ap.parse_args()
    campaign = Campaign.load()
    serializer = serializer_for(campaign.teacher)
    workers = a.workers or campaign.config.get('export_workers') or len(os.sched_getaffinity(0))
    signal.signal(signal.SIGTERM, stop)
    for kind in ['train', 'eval', 'smoke'] if a.kind == 'all' else [a.kind]:
        if kind != 'smoke' and not campaign.enabled(kind):
            continue
        result = run_once(campaign, kind, serializer, workers)
        print(kind, len(result), sum(r['tokens'] for r in result.values()), flush=True)
