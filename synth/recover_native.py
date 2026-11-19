"""Evidence-checked recovery of a finalized Pi journal tail after a write EIO.

The JSON stdout event stream is an independent structured Pi artifact, not a
text transcript. Original journals remain untouched. Recovery requires an
explicit checksummed sidecar and exact agreement on every surviving message.
"""
import copy
import hashlib
import json

from .common import read, sha


def recover_sparse_terminal(folder, path, lines, parse_errors, entries):
    """Read one reviewed NUL-prefixed EIO terminal without changing its journal.

    A failed filesystem write can leave a sparse hole before Pi's final error.
    This only makes that terminal available to the independent stdout-tail
    verifier below; it never accepts or invents the missing message payloads.
    """
    rule = read(folder / 'native-graph-recovery.json', {})
    sparse = rule.get('nul_terminal')
    if not sparse or rule.get('native_source') != str(path.relative_to(folder)):
        raise ValueError(f'Malformed native session {path}: lines {parse_errors}')
    if (rule.get('schema') != 'pi-eio-tail-recovery-v1'
            or not (folder / 'worker-result.json').is_file()):
        raise ValueError('Sparse terminal recovery requires an explicit finalized EIO review')
    if (sha(path) != rule['native_sha256']
            or sha(folder / rule['pi_output']) != rule['pi_output_sha256']):
        raise ValueError('Sparse terminal recovery evidence checksum changed')
    if parse_errors != [len(lines)] or sparse['line'] != len(lines):
        raise ValueError('Sparse recovery only supports one final malformed line')
    line = lines[-1]
    terminal_text = line.lstrip('\0')
    nul_bytes = len(line) - len(terminal_text)
    if (nul_bytes < 1 or nul_bytes != sparse['nul_bytes']
            or '\0' in terminal_text or not line.endswith('\n')
            or hashlib.sha256(line.encode()).hexdigest() != sparse['line_sha256']):
        raise ValueError('Sparse terminal bytes disagree with reviewed evidence')
    terminal = json.loads(terminal_text)
    message = terminal.get('message', {})
    if (terminal.get('type') != 'message' or message.get('role') != 'assistant'
            or message.get('stopReason') != 'error'
            or 'EIO:' not in message.get('errorMessage', '')):
        raise ValueError('Sparse recovery requires a terminal assistant EIO error')
    return entries + [terminal], {**sparse,
        'policy': 'Only reviewed leading NUL bytes stripped for parsing; raw journal unchanged. Missing content must match independent structured Pi stdout.'}


def recover_tail(folder, path, entries):
    sidecar = folder / 'native-graph-recovery.json'
    rule = read(sidecar)
    if not rule or rule.get('native_source') != str(path.relative_to(folder)):
        return entries, None
    if rule.get('schema') == 'pi-wire-gap-recovery-v2':
        return entries, None  # Independently verified by recover_wire_gap.
    if rule.get('schema') != 'pi-eio-tail-recovery-v1' or not (folder / 'worker-result.json').is_file():
        raise ValueError('Native tail recovery requires an explicit finalized EIO review')
    output = folder / rule['pi_output']
    if sha(path) != rule['native_sha256'] or sha(output) != rule['pi_output_sha256']:
        raise ValueError('Native tail recovery evidence checksum changed')
    ids = {e.get('id') for e in entries}
    missing = [e for e in entries if e.get('parentId') and e['parentId'] not in ids]
    final = entries[-1]
    msg = final.get('message', {})
    if (missing != [final] or final.get('type') != 'message' or msg.get('role') != 'assistant'
            or msg.get('stopReason') != 'error' or 'EIO:' not in msg.get('errorMessage', '')):
        raise ValueError('Recovery only supports one terminal missing-parent EIO event')
    events, completed = [], []
    with output.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            try:
                event = json.loads(line)
            except ValueError:
                raise ValueError(f'Malformed Pi recovery event at line {line_number}')
            events.append((line_number, event))
            if event.get('type') == 'message_end':
                completed.append((line_number, event['message']))
    native = [e['message'] for e in entries if e.get('type') == 'message']
    prefix = len(native) - 1
    if (prefix < 1 or len(completed) <= len(native)
            or [m for _, m in completed[:prefix]] != native[:-1]
            or completed[-1][1] != native[-1]):
        raise ValueError('Pi recovery stream disagrees with surviving native messages')
    start, stop = completed[prefix-1][0], completed[-1][0]
    if any(e.get('type') in ('session', 'entry_appended', 'compaction_start', 'compaction_end')
           for n, e in events if start < n < stop):
        raise ValueError('Pi recovery gap contains a graph boundary')
    recovered = []
    parent = entries[-2]['id']
    for n, message in completed[prefix:-1]:
        if message.get('role') not in ('assistant', 'toolResult'):
            raise ValueError('Unexpected role in Pi recovery tail')
        eid = f'recovered-stdout-line-{n}'
        if eid in ids:
            raise ValueError('Recovered event ID collision')
        recovered.append({'type': 'message', 'id': eid, 'parentId': parent,
            'message': message, 'recovery_source_line': n,
            'original_native_event_id': None,
            'parent_link_provenance': 'Inferred linear order from structured Pi message_end events; native event IDs were not persisted.'})
        parent = eid
    terminal = {**final, 'parentId': parent, 'original_parentId': final['parentId'],
        'parent_link_provenance': 'Reconnected through the recovered structured event tail; original parent reference preserved.'}
    provenance = {**rule, 'sidecar': str(sidecar.relative_to(folder)), 'sidecar_sha256': sha(sidecar),
        'recovered_messages': len(recovered), 'original_missing_parent': final['parentId'],
        'original_native_entries': copy.deepcopy(entries),
        'content_policy': 'Exact structured message_end payloads, once each; no text parsing or generated replacement content.',
        'graph_policy': 'Synthetic recovered IDs and inferred linear tail links are explicit; original source and terminal parent preserved.'}
    return entries[:-1] + recovered + [terminal], provenance


def canonical_wire(message):
    """Compare served payloads under the inline-<think> history adapter.

    Wire-gap recovery is only defined for teachers whose proxy inlines
    reasoning (adapter inline_reasoning_history, e.g. Nemotron)."""
    m = copy.deepcopy(message)
    content = m.get('content') or ''
    if isinstance(content, list):
        if any(p.get('type') != 'text' for p in content):
            raise ValueError('Wire gap recovery only supports text histories')
        content = ''.join(p['text'] for p in content)
    if not isinstance(content, str):
        raise ValueError('Invalid wire content')
    reasoning = m.get('reasoning_content') or m.get('reasoning')
    if m['role'] == 'assistant' and reasoning and reasoning.strip():
        content = '<think>\n' + reasoning + '</think>' + content
    result = {'role': m['role'], 'content': content}
    if m['role'] == 'tool':
        result['tool_call_id'] = m['tool_call_id']
    if m.get('tool_calls'):
        calls = []
        for c in m['tool_calls']:
            args = c['function']['arguments']
            args = json.loads(args) if isinstance(args, str) else args
            if c.get('type') != 'function' or not isinstance(args, dict):
                raise ValueError('Unsupported wire tool call')
            calls.append({'id': c['id'], 'type': 'function',
                'function': {'name': c['function']['name'], 'arguments': args}})
        result['tool_calls'] = calls
    return result


def native_from_wire(message, calls):
    """Keep exposed payloads; unknown native metadata and assistant loss stay absent/zero."""
    m = canonical_wire(message)
    if m['role'] == 'tool':
        return {'role': 'toolResult', 'toolCallId': m['tool_call_id'],
            'toolName': calls[m['tool_call_id']], 'content': [{'type': 'text', 'text': m['content']}]}
    if m['role'] != 'assistant' or not m.get('tool_calls'):
        raise ValueError('Wire gap supports only completed tool exchanges')
    content, parts = m['content'], []
    if content.startswith('<think>\n'):
        if content.count('</think>') != 1 or content.count('<think>') != 1:
            raise ValueError('Ambiguous reasoning wrapper in wire recovery')
        reasoning, content = content[len('<think>\n'):].split('</think>', 1)
        parts.append({'type': 'thinking', 'thinking': reasoning})
    if content:
        parts.append({'type': 'text', 'text': content})
    for call in m['tool_calls']:
        calls[call['id']] = call['function']['name']
        parts.append({'type': 'toolCall', 'id': call['id'], 'name': call['function']['name'],
            'arguments': call['function']['arguments']})
    return {'role': 'assistant', 'content': parts, 'stopReason': 'unknown', 'sft_loss_weight': 0}


def recover_wire_gap(folder, path, entries, convert, calls):
    rule = read(folder/'native-graph-recovery.json', {})
    if rule.get('schema') != 'pi-wire-gap-recovery-v2' or rule.get('native_source') != str(path.relative_to(folder)):
        return entries, None
    if not (folder/'worker-result.json').is_file():
        raise ValueError('Wire gap recovery requires a finalized review')
    if sha(path) != rule['native_sha256'] or sha(folder/rule['pi_output']) != rule['pi_output_sha256']:
        raise ValueError('Wire gap evidence checksum changed')
    if sha(folder/rule['wire_log']) != rule['wire_log_sha256']:
        raise ValueError('Wire gap wire log checksum changed')
    by_id = {c['id']: c for c in calls}
    if len(rule['wire_calls']) != 2:
        raise ValueError('Wire gap requires two independent request snapshots')
    requests = [by_id[i]['body'] for i in rule['wire_calls']]
    left, right = requests
    if (len(left['messages']) >= len(right['messages'])
            or left['messages'] != right['messages'][:len(left['messages'])]
            or not left.get('tools') or left['tools'] != right.get('tools')):
        raise ValueError('Wire gap snapshots disagree on the complete history prefix or schemas')
    ids = {e.get('id') for e in entries}
    missing = [e for e in entries if e.get('parentId') and e['parentId'] not in ids]
    if (len(missing) != 1 or missing[0].get('id') != rule['successor_id']
            or missing[0]['parentId'] != rule['missing_parent_id']):
        raise ValueError('Wire gap review does not match the one missing native parent')
    at = entries.index(missing[0])
    if entries[at-1].get('id') != rule['predecessor_id']:
        raise ValueError('Wire gap predecessor changed')
    if any(e.get('type') in ('compaction', 'branch_summary') for e in entries):
        raise ValueError('Wire gap cannot cross compaction')
    before = [canonical_wire(convert(e['message'])) for e in entries[:at] if e.get('type') == 'message']
    after = [canonical_wire(convert(e['message'])) for e in entries[at:] if e.get('type') == 'message']
    wire = [canonical_wire(m) for m in left['messages']]
    start, stop = rule['wire_gap_start'], rule['wire_gap_stop']
    if (start != len(before) or wire[:start] != before or not start < stop < len(wire)
            or len(wire)-stop < 2 or wire[stop:] != after[:len(wire)-stop]):
        raise ValueError('Wire gap snapshots disagree with surviving native prefix or suffix')
    calls = {c['id']: c['function']['name'] for m in wire[:start] for c in m.get('tool_calls', [])}
    recovered, parent = [], entries[at-1]['id']
    for i in range(start, stop):
        eid = f'recovered-wire-message-{i}'
        if eid in ids:
            raise ValueError('Wire recovery event ID collision')
        message = native_from_wire(wire[i], calls)
        if canonical_wire(convert(message)) != wire[i]:
            raise ValueError('Wire recovery round trip changed an exposed message')
        recovered.append({'type': 'message', 'id': eid, 'parentId': parent, 'message': message,
            'sft_loss_weight': 0, 'original_native_event_id': None, 'wire_message_index': i,
            'native_metadata_unavailable': ['timestamp', 'usage', 'stopReason', 'thinkingSignature'],
            'parent_link_provenance': 'Inferred linear order from two identical served request prefixes.'})
        parent = eid
    successor = {**entries[at], 'parentId': parent, 'original_parentId': entries[at]['parentId']}
    sidecar = folder/'native-graph-recovery.json'
    provenance = {**rule, 'sidecar': str(sidecar.relative_to(folder)), 'sidecar_sha256': sha(sidecar),
        'original_native_entries': copy.deepcopy(entries), 'recovered_messages': len(recovered),
        'original_wire_gap_messages': left['messages'][start:stop],
        'content_policy': 'Exact served payloads, verified in two request prefixes and against all surviving native prefix/suffix messages. Unknown native metadata is not invented; recovered assistants have zero loss.',
        'graph_policy': 'Synthetic IDs and inferred linear links explicit; original missing-parent reference and journal preserved.'}
    return entries[:at] + recovered + [successor] + entries[at+1:], provenance
