"""Run-dir adapters: find the finished attempts of a run and read each one into a Trace.

One adapter per layout (aa_evals/README.md contract plus each benchmark's own transcript files):

  pi_eval      TB 2.1, TB-Sci, TB-Sci-lite: attempts/<key>/{attempt,outcome}.json, harbor.log, trial.tar.gz
  hle          transcripts/<item>.r<k>.json (generation + judgment; the question is not stored)
  omniscience  transcripts/<item>.r<k>.json (item, model{messages, answer, ...}, judge)
  gdppdf       generations.jsonl (one row per item x repeat; grades.jsonl is the grader's, out of scope)
  critpt       transcripts.jsonl (key = <item>/r<k>/a<n>; history = the conversation)
  automationbench  transcripts.jsonl (messages, debug, perf)
  generic      items.jsonl rows whose `transcript` points at a JSON file or `<file>.jsonl#<key>`; the transcript
               holds OpenAI-style `messages` (or `history`) and/or a single response (content, reasoning,
               finish_reason, usage, errors), plus optional `events` (harness lines). New benchmarks
               (GDPval-AA, SciCode-Verified) work through this one; see README.md.

`finished()` lists keys whose attempt is over (its outcome/transcript is written). `load(key)` builds the Trace.
"""
from __future__ import annotations

import glob
import json
import os
import re
import tarfile
from pathlib import Path

from aa_evals.tracejudge.trace import Event, Trace, Turn
from aa_evals.tracejudge.jsonl import physical_lines

INTERESTING = re.compile(
    r'memcap|oom|out of memory|killed|kill\(|signal|exit (code )?-?\d+|exit -?\d+|error|exception|traceback|'
    r'timeout|timed out|refused|reset by peer|unreachable|no space|read-only|resource temporarily|'
    r'cannot allocate|too many open|proxy|50[0-9]\b|retry|retrying|stopp|cutoff|cut\b|abort', re.I)


# harness lines present in every attempt, unrelated to its outcome
HARBOR_NOISE = (
    'ValueError: too many values to unpack (expected 2)',   # harbor's results-table printer, after the trial ended
)


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _jsonl(path):
    from .hook import _warn_once
    out = []
    try:
        with open(path, errors='replace') as f:
            lines = [x.removesuffix('\r') for x in f.read().split('\n')]   # '\n' only; last = '' or unterminated
    except OSError:
        return out
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError as e:
                if i < len(lines):   # the unterminated last line is a row being written right now
                    _warn_once(path, i, e)
    return out


def _items_index(run_dir: Path) -> dict:
    """items.jsonl rows by (item_id, repeat) and by key; the last row wins (resumed runs append)."""
    idx = {}
    for r in _jsonl(run_dir / 'items.jsonl'):
        idx[(str(r.get('item_id')), r.get('repeat'))] = r
        if r.get('key'):
            idx[r['key']] = r
    return idx


def _content_text(content) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get('type') in ('image_url', 'image', 'input_image'):
                    parts.append('[image]')
                else:
                    parts.append(str(c.get('text') or c.get('content') or ''))
            else:
                parts.append(str(c))
        return '\n'.join(p for p in parts if p)
    return str(content)


PYTEST_HEADER = 'pytest summary: '
RAISED_AT = re.compile(r'^(\S+\.py):\d+: (\w+(?:Error|Exception|Exit)|AssertionError)$')
PYTEST_FINAL = re.compile(r'^=+ (.*\b(?:passed|failed|error|errors|skipped|no tests ran)\b.*?) in [\d.]+s.*=+$')
CAPTURED = re.compile(r'^-+ Captured (log|stderr|stdout) (call|setup|teardown) -+$')


def drop_captured_logs(lines: list[str]) -> list[str]:
    """Drop pytest's "Captured log ..." sections (library logging such as urllib3 retries during a Selenium teardown):
    noise that the error-keyword filter would otherwise keep over the test summary."""
    out, skip = [], False
    for x in lines:
        if CAPTURED.match(x):
            skip = 'Captured log' in x
            if skip:
                continue
        elif skip and re.match(r'^(-{5,}|={5,})', x):
            skip = False
        if not skip:
            out.append(x)
    return out


def pytest_summary(text: str) -> dict | None:
    """Counts from pytest's final line plus the exception type of each failure in "short test summary info"."""
    lines = text.splitlines()
    final = next((m.group(1) for x in reversed(lines) for m in [PYTEST_FINAL.match(x.strip())] if m), None)
    if not final:
        return None
    counts = {k: int(n) for n, k in re.findall(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)', final)}
    counts['errors'] = counts.pop('error', 0) + counts.pop('errors', 0)
    fails = []
    if 'short test summary info' in text:
        for x in text.split('short test summary info', 1)[1].splitlines():
            m = re.match(r'^(FAILED|ERROR) \S+(?: - (\w[\w.]*))?', x)
            if m:
                fails.append(f"{m.group(1)}:{m.group(2) or '?'}")
    # pytest's short-traceback location of each failure: "path.py:LINE: ExceptionType" (the raising frame)
    raised = [f'{m.group(1)}:{m.group(2)}' for x in lines for m in [RAISED_AT.match(x.strip())] if m]
    return {'passed': counts.get('passed', 0), 'failed': counts.get('failed', 0), 'errors': counts['errors'],
            'no_tests': 'no tests ran' in final, 'failures': fails[:20], 'raised_at': raised[:20]}


CRASH = re.compile(r'Fatal Python error|Segmentation fault|Bus error|core dumped|Aborted|SIGSEGV|SIGABRT')
FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)(?: in (\S+))?')


def crash_frames(text: str, n: int = 4) -> list[str]:
    """The innermost frames of a hard crash's faulthandler dump ("Current thread ... (most recent call first):")."""
    lines = text.splitlines()
    for i, x in enumerate(lines):
        if CRASH.search(x):
            for j in range(i, min(i + 6, len(lines))):
                if 'most recent call first' in lines[j]:
                    return [y.strip() for y in lines[j + 1:j + 1 + 3 * n] if FRAME.match(y)][:n]
            return []
    return []


def openai_turns(messages, start=0, ref_prefix='msg') -> list[Turn]:
    """OpenAI chat messages (system/user/assistant with tool_calls/tool) -> Turns."""
    turns, i = [], start
    for n, m in enumerate(messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get('role', '?')
        ref = f'{ref_prefix}:{n}'
        if role in ('system', 'user', 'developer'):
            turns.append(Turn(i, role, 'task', _content_text(m.get('content')), ref)); i += 1
        elif role == 'assistant' and isinstance(m.get('blocks'), list):
            # Stirrup (CritPt): blocks of kind reasoning / text / tool_call
            usage = m.get('token_usage') if isinstance(m.get('token_usage'), dict) else {}
            for b in m['blocks']:
                k = (b or {}).get('kind')
                if k == 'reasoning':
                    turns.append(Turn(i, 'assistant', 'thinking', str(b.get('content') or ''), ref))
                elif k == 'tool_call':
                    turns.append(Turn(i, 'assistant', 'tool_call', f"{b.get('name')}: {b.get('arguments')}", ref))
                else:
                    turns.append(Turn(i, 'assistant', 'text', str(b.get('text') or b.get('content') or ''), ref))
                i += 1
            if turns and usage:
                turns[-1].meta['token_usage'] = usage
            if not m['blocks']:
                turns.append(Turn(i, 'assistant', 'text', '', ref)); i += 1
        elif role == 'assistant':
            rc = m.get('reasoning_content') or m.get('reasoning')
            if rc:
                turns.append(Turn(i, 'assistant', 'thinking', str(rc), ref)); i += 1
            txt = _content_text(m.get('content'))
            if txt or not m.get('tool_calls'):
                turns.append(Turn(i, 'assistant', 'text', txt, ref,
                                  {k: m.get(k) for k in ('finish_reason', 'stop_reason') if m.get(k)})); i += 1
            for tc in m.get('tool_calls') or []:
                # OpenAI nests name/arguments under 'function'; AutomationBench (verifiers) stores them flat.
                fn = (tc or {}).get('function') or tc or {}
                turns.append(Turn(i, 'assistant', 'tool_call', f"{fn.get('name')}: {fn.get('arguments')}", ref)); i += 1
        elif role in ('tool', 'function', 'toolResult'):
            turns.append(Turn(i, 'tool', 'tool_result', _content_text(m.get('content')), ref,
                              {'name': m.get('name') or m.get('toolName')})); i += 1
        else:
            turns.append(Turn(i, role, 'text', _content_text(m.get('content')), ref)); i += 1
    return turns


def single_turns(resp: dict, task: str | None = None, ref='response') -> list[Turn]:
    """A single response block (content/answer, reasoning, finish_reason, usage) -> Turns."""
    turns = []
    if task:
        turns.append(Turn(0, 'user', 'task', task, 'request'))
    reasoning = resp.get('reasoning') or resp.get('reasoning_content')
    if reasoning:
        turns.append(Turn(len(turns), 'assistant', 'thinking', str(reasoning), ref))
    content = resp.get('content') if resp.get('content') is not None else resp.get('answer')
    usage = resp.get('usage') or {}
    turns.append(Turn(len(turns), 'assistant', 'text', '' if content is None else str(content), ref,
                      {'finish_reason': resp.get('finish_reason'),
                       'completion_tokens': usage.get('completion_tokens'),
                       'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                       'prompt_tokens': usage.get('prompt_tokens')}))
    return turns


def _attempt_events(attempts, source='client') -> list[Event]:
    ev = []
    for a in attempts or []:
        if isinstance(a, dict) and (a.get('status') not in (None, 'ok', 200) or a.get('error')):
            ev.append(Event(source, json.dumps(a)[:400], f'{source}:attempt {a.get("attempt")}'))
    return ev


def gen_max_tokens(config: dict):
    """The model's max_tokens in a run config (generation.*, else limits.*, else top level; never the judge's)."""
    for d in (config.get('generation'), config.get('limits'), config):
        if isinstance(d, dict) and isinstance(d.get('max_tokens'), int):
            return d['max_tokens']
    return None


class Layout:
    name = 'base'

    def __init__(self, run_dir: Path, bench: str):
        self.run_dir, self.bench = Path(run_dir), bench
        self.config = _read_json(self.run_dir / 'config.json', {}) or {}

    def finished(self) -> list[str]:
        raise NotImplementedError

    def load(self, key: str) -> Trace:
        raise NotImplementedError

    def complete(self) -> bool:
        """The run has finished (its summary says so); watch stops once every finished attempt is judged."""
        s = _read_json(self.run_dir / 'summary.json', {}) or {}
        return bool(s) and s.get('complete', True) is not False and not s.get('n_pending')


# ----------------------------------------------------------------------------------------------- pi_eval (TB)
AGENT_COMMS = ('pi', 'node')


def _offline_proxy_review(run_dir, key, outcome, config):
    """Resolve only a root-written, hash-bound public-index review; never an LLM verdict."""
    if outcome.get('offline_proxy_review_required') is not True:
        return {}
    if any(isinstance(item, dict) and item.get('reason') == 'agent_tool_network_origin_not_proven'
           for item in outcome.get('offline_proxy_review_evidence') or []):
        from .tool_network_review import review_tool_network
        return review_tool_network(run_dir, key, outcome, config)
    import hashlib
    def sha(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('review artifact must be a regular file')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                h.update(block)
        return h.hexdigest()
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    def proof(spec):
        if not isinstance(spec, dict) or not Path(spec.get('path', '')).is_absolute():
            raise ValueError('missing absolute source proof path')
        actual = sha(spec['path'])
        if actual != spec.get('sha256'):
            raise ValueError('source proof SHA256 mismatch')
        return actual
    evidence = outcome.get('offline_proxy_review_evidence')
    record = {'offline_proxy_review_required': True, 'offline_proxy_review_pending': True,
              'offline_proxy_review_evidence': evidence}
    path = Path(run_dir) / 'attempts' / key / 'offline-proxy-review.json'
    if not path.exists():
        return record
    try:
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('missing classifier evidence')
        receipt_sha = sha(path)
        # A single object, never append/latest-wins: duplicate JSON fields or
        # duplicate evidence approvals cannot override a previous binding.
        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError('duplicate review receipt field')
                result[name] = value
            return result
        receipt = json.loads(path.read_text(), object_pairs_hook=unique)
        version = receipt.get('schema_version') if isinstance(receipt, dict) else None
        verdict = 'expected_agent_host_denial' if version == 3 else 'expected_policy_denial'
        if (not isinstance(receipt, dict) or type(version) is not int or version not in (1, 2, 3)
                or receipt.get('key') != key or receipt.get('verdict') != verdict):
            raise ValueError('unsupported or mismatched review receipt')
        d = Path(run_dir) / 'attempts' / key
        expected = {'archive_sha256': sha(d / 'trial.tar.gz'), 'attempt_sha256': sha(d / 'attempt.json'),
                    'marker_evidence_sha256': digest(evidence)}
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise ValueError('review does not bind this immutable generation/evidence')
        source_image_sha = proof(receipt.get('source_image'))
        instruction = receipt.get('instruction') or {}
        bundle = Path(config.get('bundle', ''))
        if sha(bundle) != config.get('bundle_sha256') or instruction.get('bundle_sha256') != config.get('bundle_sha256'):
            raise ValueError('instruction task bundle binding differs')
        task = key.rsplit('__a', 1)[0]
        member = f'tasks/{task}/instruction.md'
        if instruction.get('member') != member:
            raise ValueError('instruction is not the reviewed task')
        with tarfile.open(bundle) as tf:
            instruction_sha = hashlib.sha256(tf.extractfile(member).read()).hexdigest()
        if instruction.get('sha256') != instruction_sha:
            raise ValueError('instruction SHA256 differs')
        audit_field = {1: 'index_audit', 2: 'source_audit', 3: 'host_probe_audit'}[version]
        index_audit_sha = proof(receipt.get(audit_field))
        audit = json.loads(Path(receipt[audit_field]['path']).read_text(), object_pairs_hook=unique)
        bindings = {**expected, 'source_image_sha256': source_image_sha,
                    'instruction_sha256': instruction_sha, 'bundle_sha256': config['bundle_sha256']}
        from urllib.parse import urlsplit
        import ipaddress
        host_probes = [item for item in evidence if isinstance(item, dict) and item.get('reason') == 'agent_host_egress_unresolved']
        if version == 3:
            if not host_probes:
                raise ValueError('host-probe receipt needs actual unresolved host evidence')
            relay = receipt.get('relay_config') or {}
            with tarfile.open(d / 'trial.tar.gz') as tf:
                member = tf.getmember(relay.get('member', ''))
                if not member.isfile() or Path(member.name).name != 'config.json' or member.size > 1 << 20:
                    raise ValueError('actual trial relay config required')
                raw = tf.extractfile(member).read()
            relay_sha = hashlib.sha256(raw).hexdigest()
            relay_cfg = json.loads(raw, object_pairs_hook=unique)
            bridge_url = relay_cfg['agent']['env']['OPENAI_BASE_URL']
            bridge = urlsplit(bridge_url)
            bridge_ip = ipaddress.ip_address(bridge.hostname or '')
            if (bridge.scheme != 'http' or bridge.username or bridge.password or bridge.path != '/v1'
                    or not bridge_ip.is_private or bridge_ip.is_loopback or not bridge.port
                    or relay.get('sha256') != relay_sha):
                raise ValueError('relay config is not the actual non-loopback host bridge')
            targets = []
            for item in host_probes:
                if (item.get('bridge_url') != bridge_url or item.get('bridge_config_sha256') != relay_sha
                        or item.get('known_proxy_url') != 'http://10.22.0.107:8080'
                        or not isinstance(item.get('probe_targets'), list) or not item['probe_targets']
                        or not all(isinstance(target, str) for target in item['probe_targets'])):
                    raise ValueError('host evidence differs from actual relay config/targets')
                targets.extend(item['probe_targets'])
            targets = sorted(set(targets))
            seen_bridge = seen_proxy = False
            for target in targets:
                parsed = urlsplit(target)
                hostport = (parsed.hostname, parsed.port)
                if parsed.username or parsed.password or parsed.query or parsed.fragment:
                    raise ValueError('ambiguous host probe target')
                if hostport == (bridge.hostname, bridge.port):
                    if not ((parsed.scheme == 'tcp' and not parsed.path) or (parsed.scheme == 'http' and parsed.path == '/v1/models')):
                        raise ValueError('unknown bridge probe')
                    seen_bridge = True
                elif hostport == ('10.22.0.107', 8080):
                    if parsed.scheme != 'http' or parsed.path not in ('', '/'):
                        raise ValueError('unknown proxy probe')
                    seen_proxy = True
                else:
                    try:
                        public = ipaddress.ip_address(parsed.hostname or '').is_global
                    except ValueError:
                        host = parsed.hostname or ''
                        public = bool(re.fullmatch(r'[A-Za-z0-9.-]+', host)) and '.' in host and not host.endswith(('.local', '.localhost', '.internal', '.lan', '.test', '.invalid', '.cluster.internal')) and not re.fullmatch(r'[0-9.]+', host)
                    if parsed.scheme not in ('http', 'https') or not public:
                        raise ValueError('host receipt cannot approve other private/task targets')
            if (not seen_bridge or not seen_proxy or audit.get('probe_targets') != targets
                    or audit.get('known_bridge_url') != bridge_url or audit.get('known_proxy_url') != 'http://10.22.0.107:8080'
                    or audit.get('model_requests_continued') is not True
                    or audit.get('non_task_host_probes') is not True
                    or audit.get('required_task_service_failure') is not False):
                raise ValueError('missing source-backed non-task/continued-model host-probe attestation')
            bindings.update(relay_config_sha256=relay_sha, host_probe_targets_sha256=digest(targets))
        elif host_probes:
            raise ValueError('public receipt cannot resolve an unresolved host probe')
        if (type(audit.get('schema_version')) is not int or audit.get('schema_version') != version or audit.get('key') != key
                or audit.get('verdict') != verdict
                or audit.get('bindings') != bindings
                or not isinstance(audit.get('evidence'), str) or not audit['evidence'].strip()):
            raise ValueError('index audit does not bind actual source-image/instruction/generation proof')
        managers = {'default_pip_index_not_proven': 'pip', 'pip_effective_indexes_not_proven': 'pip', 'git_effective_urls_not_proven': 'git',
                    'apt_effective_sources_not_proven': 'apt'}
        allowed_reasons = set(managers) | ({'agent_host_egress_unresolved'} if version == 3 else set())
        if any(not isinstance(item, dict) or item.get('reason') not in allowed_reasons for item in evidence):
            raise ValueError('unsupported source-review evidence')
        required = {managers[item['reason']] for item in evidence if item['reason'] in managers}
        if version == 1:
            if required != {'pip'} or audit.get('public_default_index') is not True:
                raise ValueError('legacy index receipt cannot approve other source managers')
            indexes = audit.get('indexes')
        else:
            sources = audit.get('sources')
            if (not isinstance(sources, list) or (not sources and (required or version != 3)) or any(not isinstance(item, dict) for item in sources)
                    or {item.get('manager') for item in sources} != required
                    or len(sources) != len(required)):
                raise ValueError('source audit has missing, duplicate or unexpected managers')
            indexes = []
            for item in sources:
                urls = item.get('urls')
                if (item.get('public_effective_sources') is not True or not isinstance(urls, list)
                        or not urls or not all(isinstance(url, str) for url in urls)
                        or len(urls) != len(set(urls)) or not isinstance(item.get('evidence'), str)
                        or not item['evidence'].strip()):
                    raise ValueError('effective source proof is incomplete')
                indexes.extend(urls)
        # Do not automatically bless a claimed private or mixed index in a manual receipt.
        if not isinstance(indexes, list) or (not indexes and version != 3) or len(indexes) != len(set(indexes)):
            raise ValueError('missing or duplicate source indexes')
        for url in indexes:
            parsed = urlsplit(url)
            _ = parsed.port
            host = parsed.hostname or ''
            if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password:
                raise ValueError('invalid public source index')
            try:
                public = ipaddress.ip_address(host).is_global
            except ValueError:
                public = (bool(re.fullmatch(r'[A-Za-z0-9.-]+', host)) and not re.fullmatch(r'[0-9.]+', host) and '.' in host
                          and not host.lower().rstrip('.').endswith(('.local', '.localhost', '.internal', '.lan', '.test', '.invalid', '.cluster.internal')))
            if not public:
                raise ValueError('source index is private or ambiguous')
        record.update(offline_proxy_review_pending=False, offline_proxy_review_receipt_sha256=receipt_sha,
                      offline_proxy_review_index_audit_sha256=index_audit_sha,
                      offline_proxy_review_status='resolved_expected_policy_denial')
        if version in (2, 3):
            record.pop('offline_proxy_review_index_audit_sha256', None)
            record['offline_proxy_review_source_audit_sha256' if version == 2 else 'offline_proxy_review_host_probe_audit_sha256'] = index_audit_sha
        if version == 3:
            record['offline_proxy_review_status'] = 'resolved_expected_agent_host_denial'
    except (OSError, ValueError, TypeError, KeyError, AttributeError, tarfile.TarError) as exc:
        record['offline_proxy_review_error'] = str(exc)[:400]
    return record


class PiEvalLayout(Layout):
    name = 'pi_eval'

    @staticmethod
    def detect(run_dir: Path) -> bool:
        return (run_dir / 'attempts').is_dir() and (run_dir / 'queue').is_dir()

    def finished(self):
        keys = [Path(p).parent.name for p in glob.glob(str(self.run_dir / 'attempts' / '*' / 'outcome.json'))]
        return sorted(keys)

    def _counted(self) -> dict:
        """key -> counted: per task attempt, the first retry with a non-infra outcome (else the last retry)."""
        keys = self.finished()
        if getattr(self, '_counted_for', None) == len(keys):
            return self._counted_cache
        groups = {}
        for k in keys:
            m = re.match(r'(.*__a\d+)__r(\d+)$', k)
            base, r = (m.group(1), int(m.group(2))) if m else (k, 0)
            groups.setdefault(base, []).append((r, k))
        out = {}
        for base, lst in groups.items():
            lst.sort()
            pick = None
            for r, k in lst:
                o = _read_json(self.run_dir / 'attempts' / k / 'outcome.json', {}) or {}
                if not o.get('infrastructure_failure'):
                    pick = k
                    break
            pick = pick or lst[-1][1]
            for _, k in lst:
                out[k] = (k == pick)
        self._counted_for, self._counted_cache = len(keys), out
        return out

    def load(self, key):
        d = self.run_dir / 'attempts' / key
        att = _read_json(d / 'attempt.json', {}) or {}
        o = _read_json(d / 'outcome.json', {}) or {}
        st = o.get('stats') or {}
        m = re.match(r'(.*)__a(\d+)__r(\d+)$', key)
        item, rep = (m.group(1), int(m.group(2))) if m else (att.get('task', key), att.get('attempt'))
        rec = {
            'status': 'infra_error' if o.get('infrastructure_failure') else 'ok',
            'score': o.get('score'), 'reward': o.get('reward'), 'returncode': o.get('returncode'),
            'exception_type': o.get('exception_type'), 'infrastructure_failure': o.get('infrastructure_failure'),
            'infra_reason': o.get('infrastructure_reason'), 'pi_error': o.get('pi_error'),
            'end_state': st.get('end_state'), 'end_state_agent': st.get('end_state_agent'),
            'context_limit': st.get('context_limit'), 'output_cap_hits': st.get('output_cap_hits'),
            'model_requests': st.get('model_requests'), 'wall_sec': o.get('wall_sec'),
            'cutoff': bool(o.get('cutoff') or o.get('tail_cut') or st.get('end_state') == 'worker_stopped_at_cutoff'),
            'unscored': o.get('unscored'), 'retry': att.get('retry'),
            'agent_internet': self.config.get('agent_internet', True),
            'protocol': {k: (att.get('protocol') or {}).get(k) for k in
                         ('context_window', 'max_output_tokens', 'agent_timeout_sec', 'pi_bash_default_timeout_sec')},
        }
        rec.update(_offline_proxy_review(self.run_dir, key, o, self.config))
        from .grading_replay import trace_evidence
        rec.update(trace_evidence(self.run_dir, key, o))
        if rec['agent_internet'] is False:
            # The digest omits bare False values; retain an explicit policy object for the judge too.
            rec['agent_network'] = {'mode': 'offline'}
        events = []
        if rec.get('grading_replay'):
            proof = rec['grading_replay']
            events.append(Event('grading replay',
                f"CURRENT validated native grade: score={proof['score']}; {proof['current_grading']}. "
                f"HISTORICAL evidence below: {proof['historical_failure']}. "
                f"Receipt SHA256 {proof['receipt_sha256']}; certificate {proof['certificate_sha256']}.",
                'grading-replay-review.json'))
        if rec.get('grading_replay_review_pending'):
            events.append(Event('grading replay', 'UNRESOLVED grading replay certificate: ' +
                                rec.get('grading_replay_review_error', ''), 'grading-replay-review.json'))
        if o.get('exception_message'):
            events.append(Event('outcome', f"{o.get('exception_type')}: {o['exception_message'][:600]}",
                                'outcome.json:exception_message'))
        events += self._harbor_log(d / 'harbor.log')
        turns = []
        arc = d / 'trial.tar.gz'
        if arc.exists():
            ev2, turns = self._archive(arc, o.get('trial_dir_in_archive'))
            events += ev2
        counted = self._counted().get(key, True)
        return Trace(self.bench, key, item, rep, 'agentic', rec, events, turns, counted, str(d))

    @staticmethod
    def _harbor_log(path: Path) -> list[Event]:
        try:
            lines = path.read_text(errors='replace').splitlines()
        except OSError:
            return []
        ev, seen = [], set()
        for n, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s[0] in '│╭╰─' or 'Mean:' in s:
                continue   # rich traceback frames and progress bars
            if INTERESTING.search(s) and s not in seen and not any(x in s for x in HARBOR_NOISE):
                seen.add(s)
                ev.append(Event('harbor.log', s[:400], f'harbor.log:{n}'))
        return ev[:60]

    def _archive(self, arc: Path, trial_rel: str | None):
        events, turns, pending_call = [], [], None
        try:
            tf = tarfile.open(arc)
            members = tf.getmembers()
        except (OSError, tarfile.TarError) as e:
            return [Event('archive', f'unreadable trial archive: {e!r}', str(arc))], []

        def text(m, limit=None):
            try:
                b = tf.extractfile(m).read()
            except (OSError, tarfile.TarError, AttributeError):
                return ''
            s = b.decode(errors='replace')
            return s if limit is None else s[-limit:]

        for m in members:
            nm = m.name
            base = nm.rsplit('/', 1)[-1]
            if '/sessions/' in nm and nm.endswith('.jsonl'):
                turns += self._session(text(m), len(turns))
            elif base == 'exception.txt':
                s = text(m)
                inflight = _inflight_call(s)
                if inflight:
                    pending_call = inflight
                # the last frames and the error line; the middle is harbor's own stack
                tail = '\n'.join(x for x in s.splitlines()[-6:] if x.strip())
                events.append(Event('exception.txt', tail[-800:], 'trial/exception.txt'))
            elif base == 'proxy.log':
                for n, line in enumerate(text(m).splitlines(), 1):
                    if line.strip() and not line.startswith(('====', '(Press')):
                        events.append(Event('proxy.log', line[:300], f'pi-cont16k/proxy.log:{n}'))
            elif base == 'context-guard.log':
                for n, line in enumerate(text(m).splitlines()[-8:], 1):
                    if line.strip():
                        events.append(Event('context-guard.log', line[:300], 'pi-cont16k/context-guard.log'))
            elif nm.endswith('verifier/test-stdout.txt'):
                s = text(m)
                lines = drop_captured_logs(s.splitlines())
                keep = [x for x in lines if re.search(r'error|Error|ERROR|FAILED|PASSED|passed|failed|Traceback|'
                                                      r'not found|No such file|Killed|timeout', x)]
                body = '\n'.join((keep[:12] + ['…'] + keep[-12:]) if len(keep) > 24 else keep)[-2500:]
                ps = pytest_summary(s)
                if ps:   # pytest's own verdict first: did the tests run, and how did they fail?
                    body = PYTEST_HEADER + json.dumps(ps) + '\n' + body
                frames = crash_frames(s)
                if frames:   # where a hard crash (segfault, abort) happened: the model's solution or the test harness
                    body += '\ncrash frames (innermost first):\n' + '\n'.join(frames)
                events.append(Event('verifier', body, 'verifier/test-stdout.txt'))
            elif nm.endswith('agent/env_clock.json'):
                c = _read_json_text(text(m))
                if c:
                    events.append(Event('env_clock', json.dumps(c)[:300], 'agent/env_clock.json'))
            elif base == 'result.json' and trial_rel and nm.rstrip('/').endswith(trial_rel.rstrip('/') + '/result.json'):
                r = _read_json_text(text(m)) or {}
                vr = r.get('verifier_result') or {}
                ex = r.get('exception_info') or {}
                events.append(Event('result.json', json.dumps({'rewards': vr.get('rewards'),
                                                               'exception': ex.get('exception_type')})[:300],
                                    'trial/result.json'))
        if not turns:
            events.append(Event('archive', 'no Pi session in the trial (the agent never produced a turn)', str(arc)))
        elif pending_call:
            last = next((x for x in reversed(turns) if x.kind == 'tool_call'), None)
            if last is None or pending_call not in last.text:
                turns.append(Turn(len(turns), 'assistant', 'tool_call', f'bash: {pending_call}', 'exception.txt (in flight when the agent ended)',
                                  {'in_flight': True}))
        return events, turns

    @staticmethod
    def _session(s: str, start: int) -> list[Turn]:
        turns, i = [], start
        n = 0
        for line in physical_lines(s):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get('type') != 'message':
                continue
            n += 1
            msg = ev.get('message') or {}
            role = msg.get('role')
            ref = f'turn {n}'
            content = msg.get('content')
            if role == 'user':
                turns.append(Turn(i, 'user', 'task', _content_text(content), ref)); i += 1
            elif role == 'assistant':
                for c in content if isinstance(content, list) else [{'type': 'text', 'text': content}]:
                    t = c.get('type')
                    if t == 'thinking':
                        turns.append(Turn(i, 'assistant', 'thinking', c.get('thinking') or '', ref))
                    elif t == 'toolCall':
                        a = c.get('arguments')
                        a = a.get('command', a) if isinstance(a, dict) else a
                        turns.append(Turn(i, 'assistant', 'tool_call', f"{c.get('name')}: {a}", ref))
                    else:
                        turns.append(Turn(i, 'assistant', 'text', str(c.get('text') or ''), ref))
                    i += 1
                meta = {'stop_reason': msg.get('stopReason')}
                if msg.get('errorMessage'):
                    meta['error'] = str(msg['errorMessage'])[:500]
                if turns and turns[-1].role == 'assistant':
                    turns[-1].meta.update(meta)
                if msg.get('stopReason') == 'error':
                    turns.append(Turn(i, 'assistant', 'error', str(msg.get('errorMessage') or 'stopReason=error'), ref)); i += 1
            elif role == 'toolResult':
                turns.append(Turn(i, 'tool', 'tool_result', _content_text(content), ref,
                                  {'is_error': msg.get('isError'), 'name': msg.get('toolName'),
                                   'duration_ms': (msg.get('details') or {}).get('durationMs') or msg.get('durationMs')}))
                i += 1
        return turns


def _inflight_call(s: str) -> str | None:
    """The last bash command in Pi's stdout events (exception.txt carries their tail): the call running, or just
    run, when the agent ended. The session file only holds finished messages, so it can miss this one."""
    last = None
    for m in re.finditer(r'"(?:args|arguments)":\{"command":"((?:[^"\\]|\\.)*)"', s):
        last = m.group(1)
    if last is None:
        return None
    try:
        return json.loads(f'"{last}"')
    except ValueError:
        return last


def _read_json_text(s):
    try:
        return json.loads(s)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------- per-file transcripts
class TranscriptDirLayout(Layout):
    """transcripts/<item>.r<k>.json (HLE, Omniscience)."""

    @staticmethod
    def detect(run_dir: Path) -> bool:
        return (run_dir / 'transcripts').is_dir() and bool(glob.glob(str(run_dir / 'transcripts' / '*.r*.json')))

    def finished(self):
        return sorted(Path(p).name[:-5] for p in glob.glob(str(self.run_dir / 'transcripts' / '*.r*.json')))

    def _split(self, key):
        item, _, rep = key.rpartition('.r')
        return item, int(rep) if rep.isdigit() else None


class HLELayout(TranscriptDirLayout):
    name = 'hle'

    def load(self, key):
        item, rep = self._split(key)
        t = _read_json(self.run_dir / 'transcripts' / f'{key}.json', {}) or {}
        g = t.get('generation') or {}
        row = _items_index(self.run_dir).get((item, rep), {})
        rec = {k: row.get(k) for k in ('status', 'score', 'error_stage', 'error', 'finish_reason', 'stop_reason',
                                       'reasoning_unclosed', 'model_error_kind', 'model_attempts')}
        rec['status'] = rec.get('status') or g.get('status')
        rec['finish_reason'] = rec.get('finish_reason') or g.get('finish_reason')
        j = t.get('judgment') or {}
        if rec.get('score') is None and j.get('correct') in ('yes', 'no'):   # online: items.jsonl comes at the end
            rec['score'] = 1 if j['correct'] == 'yes' else 0
        rec['max_tokens'] = g.get('max_tokens') or gen_max_tokens(self.config)   # per request under a window cap
        image_inputs = self.config.get('image_inputs')
        if isinstance(image_inputs, dict) and isinstance(image_inputs.get('sent'), bool):
            rec['image_inputs'] = image_inputs
            if image_inputs['sent'] is False:
                rec['input_modality_policy'] = (
                    'This run explicitly uses text-only generation, including questions that refer to figures. '
                    'Images are intentionally omitted by this recorded protocol. '
                    'This does not establish whether an image-enabled run delivered its inputs correctly.')
        events = _attempt_events(g.get('attempts'))
        if g.get('error') and g.get('status') == 'infra_error':
            events.append(Event('client', str(g.get('error'))[:500], 'generation.error'))
        elif g.get('error'):   # "no final answer (finish_reason=...)": the pipeline's model-side verdict
            rec['generation_error'] = str(g.get('error'))[:200]
        # the question is never stored (HLE may not be redistributed); the judge reads only the response
        turns = single_turns(g)
        return Trace(self.bench, key, item, rep, 'single', rec, events, turns, True, f'transcripts/{key}.json')


class OmniscienceLayout(TranscriptDirLayout):
    name = 'omniscience'

    def load(self, key):
        item, rep = self._split(key)
        t = _read_json(self.run_dir / 'transcripts' / f'{key}.json', {}) or {}
        m = t.get('model') or {}
        row = _items_index(self.run_dir).get((item, rep), {})
        rec = {k: row.get(k) for k in ('status', 'score', 'grade', 'finish_reason', 'truncated', 'error')}
        rec['status'] = rec.get('status') or ('ok' if m.get('answer') is not None else None)
        rec['max_tokens'] = gen_max_tokens(self.config)
        grade = (t.get('judge') or {}).get('grade')
        if rec.get('score') is None and grade:   # online: items.jsonl comes at the end
            rec['score'] = 1 if grade == 'CORRECT' else 0
        events = _attempt_events(m.get('attempts') if isinstance(m.get('attempts'), list) else None)
        for e in m.get('errors') or []:
            events.append(Event('client', str(e)[:400], 'model.errors'))
        q = (t.get('item') or {}).get('question')
        turns = single_turns({**m, 'content': m.get('answer')}, task=q)
        return Trace(self.bench, key, item, rep, 'single', rec, events, turns, True, f'transcripts/{key}.json')


# -------------------------------------------------------------------------------------------- jsonl transcripts
class JsonlLayout(Layout):
    """One transcript row per attempt in one jsonl file; the last row of a key wins."""
    file = 'transcripts.jsonl'

    def _rows(self) -> dict:
        path = self.run_dir / self.file
        try:
            st = path.stat()
        except OSError:
            return {}
        sig = (st.st_size, st.st_mtime)
        if getattr(self, '_sig', None) != sig:
            rows = {}
            for r in _jsonl(path):
                rows[self.row_key(r)] = r
            self._cache, self._sig = rows, sig
        return self._cache

    def finished(self):
        return sorted(self._rows())

    def row_key(self, r):
        return f"{r.get('item_id')}/r{r.get('repeat')}"


class GDPpdfLayout(JsonlLayout):
    name = 'gdppdf'
    file = 'generations.jsonl'

    @staticmethod
    def detect(run_dir):
        return (run_dir / 'generations.jsonl').exists()

    def row_key(self, r):
        return f"{r.get('task_id')}/{r.get('repeat')}"

    def load(self, key):
        g = self._rows()[key]
        item, rep = g.get('task_id'), g.get('repeat')
        row = _items_index(self.run_dir).get((item, rep), {})
        plan = g.get('plan') or {}
        generation = self.config.get('generation') or {}
        if 'max_tokens' in g:
            max_tokens, budget_source = g['max_tokens'], 'saved request'
        elif 'max_tokens' in generation:
            max_tokens, budget_source = generation['max_tokens'], 'generation config'
        else:
            max_tokens, budget_source = gen_max_tokens(self.config), 'legacy config'
        rec = {'max_tokens': max_tokens, 'max_tokens_source': budget_source,
               'output_reserve': generation.get('output_reserve'),
               'context_window': (self.config.get('context') or {}).get('window'),
               'status': row.get('status') or g.get('status'), 'score': row.get('score'), 'error_kind': g.get('error_kind'),
               'error': g.get('error'), 'finish_reason': g.get('finish_reason'),
               'plan': {k: plan.get(k) for k in ('mode', 'covered', 'n_images', 'reason', 'rejections', 'est_prompt_tokens')}}
        if max_tokens is None and budget_source != 'legacy config':
            rec['budget_observability'] = (
                ('The saved request sent no max_tokens cap; the endpoint allows the remaining context. '
                 if budget_source == 'saved request' else
                 'The generation config has no fixed max_tokens cap. The per-request budget was not saved '
                 'and may have been dynamically clamped. ')
                + 'output_reserve is an input-planning floor, not an output-token cap.')
        events = [Event('generation', f"plan: {json.dumps(rec['plan'])[:400]}", 'generations.jsonl:plan')]
        for r in g.get('retries') or []:
            events.append(Event('client', json.dumps(r)[:400], 'generations.jsonl:retries'))
        if g.get('error'):
            events.append(Event('client', f"{g.get('error_kind')}: {str(g.get('error'))[:500]}", 'generations.jsonl:error'))
        turns = single_turns({**g, 'content': g.get('answer')})
        return Trace(self.bench, key, item, rep, 'single', rec, events, turns, True, f'{self.file}#{key}')


# critpt run.py's model-side error prefixes (status model_error): the model's own output, not the harness
CRITPT_MODEL_ERRORS = ('output_limit', 'no_finish', 'summary_failed')


class CritPtLayout(JsonlLayout):
    name = 'critpt'

    @staticmethod
    def detect(run_dir):
        return (run_dir / 'transcripts.jsonl').exists() and run_dir.parent.name == 'critpt'

    def row_key(self, r):
        return r.get('key') or f"{r.get('item_id')}/r{r.get('repeat')}/a{r.get('attempt')}"

    def load(self, key):
        t = self._rows()[key]
        item, rep = t.get('item_id'), t.get('repeat')
        rows = [r for r in _jsonl(self.run_dir / 'items.jsonl')
                if r.get('item_id') == item and r.get('repeat') == rep and r.get('attempt') == t.get('attempt')]
        row = rows[-1] if rows else {}
        rec = {'status': row.get('status') or t.get('status'), 'score': row.get('score'), 'error': t.get('error') or row.get('error'),
               'turns': row.get('turns'), 'finish_reason': row.get('finish_reason'),
               **{k: row.get(k) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens') if row.get(k) is not None}}
        g = self.config.get('generation') or {}
        rec['context_window'] = g.get('context_window_tokens') or g.get('context_cap')   # protocol limits, for rules
        rec['max_output_tokens'] = g.get('max_output_tokens')
        rec['token_usage_scope'] = 'trajectory_aggregate'
        rec['budget_observability'] = (
            'Token totals aggregate a trajectory, not one context window. Failure totals can include '
            'omitted model/summary responses; successful rows may omit summary usage. '
            'Failed summary responses and the last accepted turn may be absent from saved history; '
            'their exact request budgets cannot be reconstructed from these totals.')
        # These are the two kernel-isolated OfflineSandbox modes, not an inference from tool failures.
        harness = self.config.get('harness')
        sandbox = harness.get('sandbox') if isinstance(harness, dict) else None
        isolation = sandbox.get('isolation') if isinstance(sandbox, dict) else None
        if isolation in ('apptainer-net-none', 'unshare-inside'):
            rec['agent_internet'] = False
            rec['agent_network'] = {'mode': 'offline', 'isolation': isolation,
                                    'source': 'config.harness.sandbox.isolation'}
        for name in ('n_commands', 'n_api_retries', 'n_summaries', 'n_cap_overshoot'):
            if name in row:
                rec[name] = row[name]
        events = []
        if rec.get('agent_internet') is False:
            events.append(Event('sandbox_policy', f'Agent tools are offline (kernel-isolated {isolation}); '
                                'this policy does not describe the host model client or evaluator network.',
                                'config.harness.sandbox.isolation'))
        err = str(rec.get('error') or '')
        # harness.run_trajectory records the typed stock ContextOverflowError as model_error.
        # The retained history is not the rejected request; this declares the native outcome only.
        boundary = re.match(r'^context_overflow: Context overflow reached the (original prompt|summarized context)(?:\s|$)', err)
        if rec['status'] == 'model_error' and t.get('status') == 'model_error' and boundary:
            rec['native_protocol_boundary'] = {
                'kind': 'context_overflow', 'status': 'model_error', 'boundary': boundary.group(1),
                'source': 'native ContextOverflowError recorded in transcripts/items',
                'request_budget': 'unknown; retained turns and trajectory totals do not reconstruct the failed request',
            }
            events.append(Event('native_protocol_boundary', err[:600], 'transcripts/items:error'))
        elif err and not err.startswith(CRITPT_MODEL_ERRORS):
            events.append(Event('harness', err[:600], 'transcripts.jsonl:error'))
        turns = []
        for h, conv in enumerate(t.get('history') or []):
            turns += openai_turns(conv, len(turns), ref_prefix=f'history[{h}]')
        if t.get('answer_code') is not None:
            turns.append(Turn(len(turns), 'assistant', 'text', f"[final answer_code]\n{t.get('answer_code')}", 'answer_code'))
        return Trace(self.bench, key, item, rep, 'agentic', rec, events, turns, True, f'transcripts.jsonl#{key}')


class AutomationBenchLayout(JsonlLayout):
    name = 'automationbench'

    @staticmethod
    def detect(run_dir):
        return (run_dir / 'transcripts.jsonl').exists() and run_dir.parent.name == 'automationbench'

    def load(self, key):
        t = self._rows()[key]
        item, rep = t.get('item_id'), t.get('repeat')
        row = _items_index(self.run_dir).get((item, rep), {})
        dbg = t.get('debug') or {}
        rec = {k: row.get(k) for k in ('score', 'stop_condition', 'truncated_turns', 'turns', 'final_state_score')}
        rec['status'] = row.get('status') or t.get('status')
        rec['finish_reasons'] = dbg.get('finish_reasons')
        rec['stop_reasons'] = dbg.get('stop_reasons')   # one per model response, including the last (no-tool) one
        rec['max_turns'] = (self.config.get('harness') or {}).get('max_turns', self.config.get('max_turns'))
        g = self.config.get('generation') or {}
        rec['context_window'] = g.get('context_window') or g.get('context_cap')
        rec['max_tokens_policy'] = g.get('max_tokens')
        rec['sampling_max_tokens'] = (g.get('sampling_args') or {}).get('max_tokens')
        rec['max_context_tokens'] = row.get('max_context_tokens')
        rec['n_cap_overshoot'] = row.get('n_cap_overshoot')
        rec['token_usage_scope'] = 'trajectory_aggregate'
        rec['budget_observability'] = (
            'Saved peak is the maximum observed prompt+completion count across requests. '
            'Exact per-request estimates and requested output budgets are not saved; '
            'a length finish alone does not identify an unexpected harness cutoff.')
        events = []
        for e in dbg.get('errors') or []:
            events.append(Event('client', str(e)[:500], 'debug.errors'))
        if dbg.get('empty_responses'):
            events.append(Event('client', f"empty responses: {json.dumps(dbg['empty_responses'])[:400]}", 'debug.empty_responses'))
        if t.get('error'):
            events.append(Event('harness', str(t['error'])[:500], 'error'))
        turns = openai_turns(t.get('messages'))
        return Trace(self.bench, key, item, rep, 'agentic', rec, events, turns, True, f'transcripts.jsonl#{key}')


# ----------------------------------------------------------------------------------------------------- generic
class GenericLayout(Layout):
    """items.jsonl rows with a `transcript` pointer (a JSON file, or `<file>.jsonl#<key>`)."""
    name = 'generic'

    @staticmethod
    def detect(run_dir):
        return (run_dir / 'items.jsonl').exists()

    def _rows(self):
        rows = {}
        for r in _jsonl(self.run_dir / 'items.jsonl'):
            k = r.get('key') or f"{r.get('item_id')}/r{r.get('repeat')}" + (f"/a{r['attempt']}" if r.get('attempt') is not None else '')
            rows[k] = r
        return rows

    def finished(self):
        return sorted(k for k, r in self._rows().items() if r.get('status') not in ('pending', None))

    def _transcript(self, ptr):
        if not ptr:
            return None
        path, _, frag = str(ptr).partition('#')
        p = Path(path)
        if not p.is_absolute():
            p = self.run_dir / p
        if frag:
            for r in _jsonl(p):
                if frag in (r.get('key'), f"{r.get('item_id')}/r{r.get('repeat')}", f"{r.get('item_id')}/{r.get('repeat')}",
                            f"{r.get('task_id')}/{r.get('repeat')}"):
                    return r
            return None
        if p.suffix == '.gz':
            import gzip
            try:
                return json.loads(gzip.open(p).read())
            except (OSError, ValueError):
                return None
        return _read_json(p)

    def load(self, key):
        row = self._rows()[key]
        t = self._transcript(row.get('transcript')) or {}
        rec = {k: row.get(k) for k in ('status', 'score', 'error', 'error_kind', 'finish_reason', 'stop_reason', 'truncated')}
        rec['agent_network'] = row.get('agent_network', self.config.get('agent_network'))
        events = [Event(str(e.get('source', 'harness')) if isinstance(e, dict) else 'harness',
                        (e.get('text') if isinstance(e, dict) else str(e))[:500], f'events[{n}]')
                  for n, e in enumerate(t.get('events') or [])]
        for k in ('error', 'errors'):
            v = t.get(k) or row.get(k)
            if v:
                events.append(Event('harness', str(v)[:600], k))
        msgs = t.get('messages') or (t.get('model') or {}).get('messages')
        hist = t.get('history')
        if msgs or hist:
            turns = openai_turns(msgs) if msgs else []
            for h, conv in enumerate(hist or []):
                turns += openai_turns(conv, len(turns), ref_prefix=f'history[{h}]')
            kind = 'agentic' if any(x.kind in ('tool_call', 'tool_result') for x in turns) or len(turns) > 3 else 'single'
            final = t.get('generation') or t.get('response')
            if isinstance(final, dict) and kind == 'single':
                turns += single_turns(final)[-2:]
        else:
            resp = t.get('generation') or t.get('response') or t.get('model') or t
            turns = single_turns(resp if isinstance(resp, dict) else {'content': resp})
            kind = 'single'
        for k in ('finish_reason',):
            rec[k] = rec.get(k) or t.get(k)
        trace = Trace(self.bench, key, str(row.get('item_id')), row.get('repeat'), kind, rec, events, turns, True,
                      str(row.get('transcript')))
        if self.bench.replace('-', '_') in ('scicode', 'scicode_verified'):
            from .scicode_context import enrich
            trace = enrich(trace, row, self.config, self.run_dir)
        return trace


class GDPvalLayout(GenericLayout):
    """Native generation outcomes are ready before pairwise scoring writes items.jsonl.

    Keys stay task_id/rN, matching GDPval's scored generic rows. Retryable infra
    outcomes are not final logical results: their transcript is replaced on retry.
    """
    name = 'gdpval'

    @staticmethod
    def detect(run_dir):
        if not (run_dir / 'attempts').is_dir():
            return False
        cfg = _read_json(run_dir / 'config.json', {}) or {}
        return run_dir.parent.name in ('gdpval', 'gdpval_aa') or \
            str(cfg.get('benchmark', '')).lower().replace('_', '-') in ('gdpval', 'gdpval-aa')

    def _rows(self):
        # A watcher loads many keys from one finished() snapshot. Avoid rescanning
        # every native outcome for each key; a newly written scored index invalidates it.
        try:
            st = (self.run_dir / 'items.jsonl').stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        if hasattr(self, '_native_rows') and self._items_stamp == stamp:
            return self._native_rows
        scored = super()._rows()
        by_item = {(str(r.get('item_id')), r.get('repeat')): r for r in scored.values()}
        rows, native_seen = {}, set()
        limits = self.config.get('limits') or {}
        max_tries = limits.get('max_infra_tries', 4) if isinstance(limits, dict) else 4
        if not isinstance(max_tries, int) or max_tries < 1:
            max_tries = 4
        for d in sorted((self.run_dir / 'attempts').glob('*')):
            m = re.fullmatch(r'(.+)__r(\d+)', d.name)
            if not d.is_dir() or not m:
                continue
            tid, repeat = m.group(1), int(m.group(2))
            key = f'{tid}/r{repeat}'
            item = scored.get(key) or by_item.get((tid, repeat)) or {}
            key = item.get('key') or key
            native_seen.add(key)
            outcome = _read_json(d / 'outcome.json', {}) or {}
            if not isinstance(outcome, dict) or outcome.get('status') not in ('ok', 'infra_error'):
                continue
            if outcome['status'] == 'infra_error' and len(list(d.glob('outcome.infra*.json'))) + 1 < max_tries:
                continue  # the worker will replace this transcript under the same logical key
            policy = item.get('agent_network', outcome.get('agent_network', self.config.get(
                'agent_network', {'mode': 'legacy-online', 'recorded': False})))
            row = {**item, 'key': key, 'item_id': tid, 'repeat': repeat,
                   'status': outcome['status'], 'error': outcome.get('error'), 'end': outcome.get('end'),
                   'score': item.get('score') if item.get('status') == outcome['status'] else None,
                   'agent_network': policy,
                   'transcript': item.get('transcript') or str((d / 'transcript.json.gz').absolute()),
                   '_outcome': outcome, '_outcome_path': str((d / 'outcome.json').absolute())}
            rows[key] = row
        # Older/exported runs with items but no corresponding native directory still
        # use the generic contract. A directory currently being retried stays hidden.
        rows.update({k: r for k, r in scored.items() if k not in native_seen})
        self._native_rows, self._items_stamp = rows, stamp
        return rows

    def finished(self):
        self.__dict__.pop('_native_rows', None)
        return super().finished()

    def load(self, key):
        if key not in self._rows():
            self.finished()  # a caller may ask for a newly completed key on an older adapter instance
        row = self._rows()[key]
        trace = super().load(key)
        outcome = row.get('_outcome')
        if isinstance(outcome, dict):
            trace.recorded.update(end_state=outcome.get('end'), wall_sec=outcome.get('wall_sec'),
                                  n_files=outcome.get('n_submitted'), turns=outcome.get('turns'))
            trace.events.append(Event('outcome',
                f"native outcome status={outcome.get('status')} end={outcome.get('end')} "
                f"submitted={outcome.get('n_submitted')} error={str(outcome.get('error') or '')[:400]}",
                row['_outcome_path']))
        # Keep the recorded evaluation limits visible when a context/turn limit ends
        # an otherwise normal generation. No inference about the verdict is made.
        harness = self.config.get('harness') or {}
        context = self.config.get('context') or {}
        model = self.config.get('model') or {}
        if isinstance(harness, dict):
            trace.recorded['max_turns'] = harness.get('max_turns')
        if isinstance(context, dict):
            trace.recorded['context_window'] = context.get('window')
        if isinstance(model, dict):
            trace.recorded['max_output_tokens'] = model.get('max_output')
        return trace

    def complete(self):
        selection = self.config.get('selection') or {}
        ids = selection.get('task_ids') if isinstance(selection, dict) else None
        repeats = self.config.get('repeats')
        if ids and isinstance(ids, list) and isinstance(repeats, int) and repeats > 0:
            expected = {(str(tid), repeat) for tid in ids for repeat in range(repeats)}
            finished = {(str(r.get('item_id')), r.get('repeat')) for r in self._rows().values()
                        if r.get('status') in ('ok', 'infra_error')}
            return expected <= finished  # trace-generating executions, independent of pairwise scoring
        return super().complete()


LAYOUTS = [PiEvalLayout, GDPpdfLayout, CritPtLayout, AutomationBenchLayout, GDPvalLayout]


KNOWN_BENCHES = ('tb21', 'tb4', 'tbsci', 'tbsci_lite', 'automationbench', 'critpt', 'omniscience', 'hle', 'gdppdf',
                 'gdpval', 'gdpval_aa', 'scicode', 'scicode_verified')


def bench_of(run_dir: Path, config: dict) -> str:
    """runs/<bench>/<run_id> names the benchmark; else the config's own name."""
    parent = Path(run_dir).parent.name
    if parent in KNOWN_BENCHES:
        return parent
    return str(config.get('benchmark') or config.get('bench') or parent)


def open_run(run_dir, layout: str | None = None) -> Layout:
    run_dir = Path(run_dir)
    cfg = _read_json(run_dir / 'config.json', {}) or {}
    bench = bench_of(run_dir, cfg)
    by_name = {c.name: c for c in LAYOUTS + [HLELayout, OmniscienceLayout, GenericLayout]}
    if layout:
        return by_name[layout](run_dir, bench)
    for c in LAYOUTS:
        if c.detect(run_dir):
            return c(run_dir, bench)
    if TranscriptDirLayout.detect(run_dir):
        return (OmniscienceLayout if 'omniscience' in bench.lower() else HLELayout)(run_dir, bench)
    if GenericLayout.detect(run_dir) or os.path.exists(run_dir / 'config.json'):
        return GenericLayout(run_dir, bench)
    raise SystemExit(f'{run_dir}: no known run layout (no attempts/, transcripts/, *.jsonl or items.jsonl)')
