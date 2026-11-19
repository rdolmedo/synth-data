"""Strict root source review for tool denials with an unproven network origin."""
import hashlib
import json
import ipaddress
from pathlib import Path
import re
import tarfile
from urllib.parse import urlsplit

from .jsonl import physical_lines


def review_tool_network(run_dir, key, outcome, config):
    evidence = outcome.get('offline_proxy_review_evidence')
    record = {'offline_proxy_review_required': True, 'offline_proxy_review_pending': True,
              'offline_proxy_review_evidence': evidence}

    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError('duplicate review field')
            result[name] = value
        return result

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def sha(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('review proof must be a regular file')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                h.update(block)
        return h.hexdigest()

    def proof(spec):
        if not isinstance(spec, dict) or not Path(spec.get('path', '')).is_absolute():
            raise ValueError('absolute proof path required')
        actual = sha(spec['path'])
        if actual != spec.get('sha256'):
            raise ValueError('proof hash differs')
        return actual

    d = Path(run_dir) / 'attempts' / key
    receipt_path = d / 'offline-proxy-review.json'
    if not receipt_path.exists():
        return record
    try:
        receipt_sha = sha(receipt_path)
        receipt = json.loads(receipt_path.read_text(), object_pairs_hook=unique)
        if (type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 4
                or receipt.get('key') != key or receipt.get('verdict') != 'expected_agent_tool_denial'):
            raise ValueError('tool-origin evidence requires schema4 source review')
        managers = {'default_pip_index_not_proven': 'pip', 'pip_effective_indexes_not_proven': 'pip',
                    'git_effective_urls_not_proven': 'git', 'apt_effective_sources_not_proven': 'apt'}
        if (not isinstance(evidence, list) or not evidence or any(not isinstance(item, dict)
                or item.get('reason') not in set(managers) | {'agent_tool_network_origin_not_proven'} for item in evidence)):
            raise ValueError('mixed review reasons need separate source review')
        bindings = {'archive_sha256': sha(d / 'trial.tar.gz'), 'attempt_sha256': sha(d / 'attempt.json'),
                    'marker_evidence_sha256': digest(evidence)}
        if any(receipt.get(k) != value for k, value in bindings.items()):
            raise ValueError('review generation binding differs')
        bindings['source_image_sha256'] = proof(receipt.get('source_image'))
        bundle = Path(config['bundle'])
        bundle_sha = sha(bundle)
        instruction = receipt.get('instruction') or {}
        task = key.rsplit('__a', 1)[0]
        member_name = f'tasks/{task}/instruction.md'
        if (bundle_sha != config.get('bundle_sha256') or instruction.get('bundle_sha256') != bundle_sha
                or instruction.get('member') != member_name):
            raise ValueError('instruction bundle differs')
        with tarfile.open(bundle) as tf:
            raw = tf.extractfile(member_name).read()
        instruction_sha = hashlib.sha256(raw).hexdigest()
        if instruction.get('sha256') != instruction_sha:
            raise ValueError('instruction hash differs')
        bindings.update(bundle_sha256=bundle_sha, instruction_sha256=instruction_sha)
        with tarfile.open(d / 'trial.tar.gz') as tf:
            def archived(spec, kind):
                name = spec.get('member', '') if isinstance(spec, dict) else ''
                matches = [member for member in tf.getmembers() if member.name == name]
                if len(matches) != 1 or not matches[0].isfile():
                    raise ValueError('unique regular archive member required')
                member = matches[0]
                if kind == 'session':
                    valid = '/agent/pi/sessions/' in '/' + name and name.endswith('.jsonl')
                    limit = 256 << 20
                else:
                    valid = Path(name).name == 'config.json'
                    limit = 1 << 20
                if not valid or member.size > limit:
                    raise ValueError('wrong archive evidence member')
                raw = tf.extractfile(member).read()
                actual = hashlib.sha256(raw).hexdigest()
                if actual != spec.get('sha256'):
                    raise ValueError('archive evidence hash differs')
                return raw, actual
            relay_raw, relay_sha = archived(receipt.get('relay_config'), 'relay')
            session_raw, session_sha = archived(receipt.get('session'), 'session')
        relay = json.loads(relay_raw, object_pairs_hook=unique)
        bridge_url = relay['agent']['env']['OPENAI_BASE_URL']
        if not isinstance(bridge_url, str) or not bridge_url:
            raise ValueError('missing primary relay URL')
        lines = physical_lines(session_raw.decode(errors='replace'))
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except ValueError:
                events.append(None)
        for item in evidence:
            if item['reason'] in managers:
                continue
            if item.get('bridge_url') != bridge_url or item.get('bridge_config_sha256') != relay_sha:
                raise ValueError('tool evidence relay differs')
            tool_indices = [index for index, line in enumerate(lines)
                            if hashlib.sha256(line.encode()).hexdigest() == item.get('tool_result_sha256')]
            if len(tool_indices) != 1:
                raise ValueError('unique linked tool observation required')
            index = tool_indices[0]
            event = events[index]
            msg = event.get('message') if isinstance(event, dict) else None
            if (not isinstance(msg, dict) or event.get('type') != 'message' or msg.get('role') != 'toolResult'
                    or msg.get('toolName') != 'bash' or msg.get('toolCallId') != item.get('tool_call_id')
                    or msg.get('errorMessage') or msg.get('stopReason') == 'error'):
                raise ValueError('tool observation is not a linked bash result')
            calls = []
            for prior in events[:index]:
                message = prior.get('message') if isinstance(prior, dict) else None
                if isinstance(message, dict) and message.get('role') == 'assistant':
                    calls.extend(block for block in message.get('content') or [] if isinstance(block, dict)
                                 and block.get('type') == 'toolCall' and block.get('id') == item.get('tool_call_id'))
            if (not calls or calls[-1].get('name') != 'bash'
                    or calls[-1].get('arguments', {}).get('command') != item.get('command')):
                raise ValueError('tool command differs')
            continuation = item.get('continuation') or {}
            later_index = continuation.get('line_index')
            if type(later_index) is not int or not index < later_index < len(lines):
                raise ValueError('later primary response required')
            later = events[later_index]
            message = later.get('message') if isinstance(later, dict) else None
            usage = message.get('usage') if isinstance(message, dict) else None
            output = usage.get('output') if isinstance(usage, dict) else None
            response = message.get('responseId') if isinstance(message, dict) else None
            if (not isinstance(message, dict) or later.get('type') != 'message'
                    or message.get('role') != 'assistant' or message.get('api') != 'openai-completions'
                    or message.get('provider') != 'harbor-endpoint' or not message.get('content')
                    or message.get('stopReason') == 'error' or message.get('errorMessage')
                    or type(output) is not int or output <= 0 or not isinstance(response, str)
                    or not response.startswith('chatcmpl-') or continuation.get('response_id') != response
                    or continuation.get('event_sha256') != hashlib.sha256(lines[later_index].encode()).hexdigest()):
                raise ValueError('continuation is not the bound paid primary response')
        bindings.update(relay_config_sha256=relay_sha, session_sha256=session_sha)
        audit_sha = proof(receipt.get('tool_network_audit'))
        audit = json.loads(Path(receipt['tool_network_audit']['path']).read_text(), object_pairs_hook=unique)
        required = {managers[item['reason']] for item in evidence if item['reason'] in managers}
        sources = audit.get('sources', [])
        if (not isinstance(sources, list) or any(not isinstance(source, dict) for source in sources)
                or len(sources) != len(required) or {source.get('manager') for source in sources} != required):
            raise ValueError('missing or duplicate manager source audit')
        for source in sources:
            urls = source.get('urls')
            if (source.get('public_effective_sources') is not True or not isinstance(urls, list) or not urls
                    or any(not isinstance(url, str) for url in urls) or len(urls) != len(set(urls))
                    or not isinstance(source.get('evidence'), str) or not source['evidence'].strip()):
                raise ValueError('incomplete effective source audit')
            for url in urls:
                parsed = urlsplit(url)
                _ = parsed.port
                host = parsed.hostname or ''
                try:
                    public = ipaddress.ip_address(host).is_global
                except ValueError:
                    public = (bool(re.fullmatch(r'[A-Za-z0-9.-]+', host)) and '.' in host
                        and not re.fullmatch(r'[0-9.]+', host) and not host.lower().rstrip('.').endswith(
                            ('.local', '.localhost', '.internal', '.lan', '.test', '.invalid', '.cluster.internal')))
                if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password or not public:
                    raise ValueError('private or ambiguous manager source')
        if (type(audit.get('schema_version')) is not int or audit['schema_version'] != 4
                or audit.get('key') != key or audit.get('verdict') != 'expected_agent_tool_denial'
                or audit.get('bindings') != bindings or not isinstance(audit.get('evidence'), str)
                or not audit['evidence'].strip()
                or any(audit.get(field) is not True for field in ('linked_agent_tool_observations_only',
                    'model_requests_continued', 'all_effective_sources_and_prior_mutations_reviewed'))
                or any(audit.get(field) is not False for field in ('primary_transport_failure',
                    'evaluator_failure', 'required_task_service_failure', 'unresolved_source_contradiction'))):
            raise ValueError('missing bound source audit of task and transport causality')
        record.update(offline_proxy_review_pending=False, offline_proxy_review_receipt_sha256=receipt_sha,
                      offline_proxy_review_source_audit_sha256=audit_sha,
                      offline_proxy_review_status='resolved_expected_agent_tool_denial')
    except (OSError, ValueError, TypeError, KeyError, AttributeError, tarfile.TarError) as exc:
        record['offline_proxy_review_error'] = str(exc)[:400]
    return record
