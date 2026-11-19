#!/usr/bin/env python3
"""CPU rollout worker: claims attempts of one eval run and runs its agent (Pi pi-cont16k, or Harbor terminus-2 for
runs with "harness": "terminus-2") on them in harbor_ns sandboxes.

  worker.py --run RUN_DIR --slots 32

The model endpoint is RUN/serve/endpoint.url (published by the serving GPU job). Each attempt runs
gen/run_harbor.sh on one task with -k 1, writing its Harbor trial directly to RUN/attempts/<key>/raw.
A finished attempt is classified (outcome.json, queue/done/<key>.json); infrastructure failures get
a same-seed retry (eval_q35/queue_state.py). SIGTERM: running attempts are stopped and retried.
A finished attempt's raw/ is then packed into RUN/attempts/<key>/raw.zip (rawpack.py; ~150 files -> 1; config
"raw_layout": "loose" keeps the loose tree), and its outcome's trial_dir points into the zip.
Per-sandbox memory cap (config "memcap": {"factor": F, "min_mb": M}, harbor_ns/memcap.py): a box above max(memory_mb x F,
M) MB (M for tasks that declare no memory_mb) loses its largest process, as under a container's memory limit, instead of
growing until HTCondor holds the whole worker (#258). Each attempt's box peak and kills go to attempts/<key>/memcap.json
and its outcome; the worker's cgroup use goes to workers/<name>.resources.jsonl every minute.
Per-sandbox disk cap (config "diskcap": {"factor": F, "min_mb": M, "tasks": {task: M}}, harbor_ns/diskcap.py; ON by
default: submit.py writes the benchmark's caps (common.DISKCAP_FAMILIES), a config without the key gets DISKCAP_DEFAULT,
"diskcap": null / false / 0 turns it off): no file in a box grows past max(storage_mb x F, M) MB (RLIMIT_FSIZE: the
writer gets EFBIG / SIGXFSZ), and a box whose overlay upper dir is above it loses the process writing the most, instead
of filling the worker's shared scratch and breaking the other slots (TB2.1-easy qemu-alpine-ssh: a 1 TiB dd). A
worker runs one attempt at a time of the config's "one_per_worker" tasks (caps that are a large share of its scratch).
The task's effective cap goes to attempt.json "diskcap"; box peaks, largest file and kills to
attempts/<key>/diskcap.json and the outcome.
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import time
import tomllib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BUNDLE, DISKCAP_DEFAULT, PI_CACHE, PI_RUNTIME_COMPAT, PROTOCOL, SEED_NAMESPACE, atomic, dp_rank, finish_env, now, parse_key, read, seed  # noqa: E402
from common import DISKCAP_WORKER_DISK_MB, diskcap_cap_mb, diskcap_config, diskcap_one_per_worker  # noqa: E402,F401
from queue_state import State  # noqa: E402
import rawpack  # noqa: E402
import serve_numerics  # noqa: E402
from rawpack import locate  # noqa: E402
import verifier_toolchain  # noqa: E402
from cluster import start_resource_log  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SCRATCH = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or f"/tmp/{os.environ.get('USER', 'u')}/q35eval")
INFRA = ('EnvironmentStart', 'EnvironmentBuild', 'EnvironmentSetup', 'AgentSetup', 'ApiConnection',
         'NetworkConnection', 'ConnectionRefused', 'ConnectError', 'Docker', 'Registry', 'ImageBuild',
         'BuildError', 'Sandbox')
# harbor_ns could not set up the pod's user namespace / bridge (a node-level fault: M5 11-15, gpu103 failed 70 of 504
# attempts in 1 h with 'pod user namespace did not appear', likely user-namespace exhaustion). Harbor reports it as a
# plain RuntimeError, so it was scored 0 (eval_q35) or became verifier_no_reward (pi_eval); its own reason lets a
# recycler / node excluder act on it (tbsched.lifecycle LOCAL_FAILURES 'userns_setup'; eval_q35/node_health.py).
SANDBOX_SETUP = re.compile(r'pod user namespace did not appear|pod bridge setup failed|new[ug]idmap failed|'
                           r'multi-uid sandboxes need a ')


def sandbox_setup_failure(result: dict) -> bool:
    """harbor_ns could not start the trial's pod (harbor_ns/sandbox.py Pod.start / multi-uid setup): a plain
    RuntimeError whose message starts with one of SANDBOX_SETUP's phrases, no reward, and the agent never ran (so no
    agent output can match)."""
    exc = result.get('exception_info') or {}
    reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
    etype, message = exc.get('exception_type'), exc.get('exception_message') or ''
    if reward is not None or result.get('agent_execution'):
        return False
    return ((etype == 'RuntimeError' and bool(SANDBOX_SETUP.match(message)))
            or (etype == 'FileNotFoundError' and '/etc/sub' in message))   # multi-uid: no /etc/subuid or subgid
# terminus-2 (litellm on this host): the request failed after Harbor's retries, i.e. serving was unreachable
LLM_INFRA = ('APIConnectionError', 'InternalServerError', 'ServiceUnavailableError', 'Timeout', 'APITimeoutError',
             'BadGatewayError', 'RateLimitError', 'APIError', 'ServerDisconnectedError', 'ClientConnectorError')
STOP = False
IDLE_EXIT_S = 300   # no claimable attempts this long: exit (orchestrate.py adds workers back); was 15 min (#258)
POLL_S = 20
# Cluster web proxy: every sandbox's internet path (worker_job.sh). On 2026-10-25 it refused or reset connections for
# hours (01-04 and 06-08 UTC): agents' apt/pip calls failed, and test.sh's bootstrap (apt curl, uv, uvx pytest) failed
# before pytest ran, scoring 0 whatever the agent did (36-64% of the attempts of the evals run then).
PROXY_FAIL = re.compile(
    r'(?:172\.22\.0\.107|proxy.internal:8080)[^\n"]{0,80}?(?:refused|reset|Couldn.t connect|Could not connect|Unable to connect|'
    r'Failed to connect|Connection failed|Transport endpoint)'
    r'|(?:refused|reset|Couldn.t connect|Could not connect|Unable to connect|Failed to connect|Connection failed)'
    r'[^\n"]{0,40}?(?:172\.22\.0\.107|proxy.internal:8080)|Cannot connect to proxy', re.I)
# tests/test.sh of every staged task. The task's own script is test_task.sh. 1) Offline: test_offline.sh
# (verifier_toolchain.py: the task's script with its uv bootstrap served from /tests/.toolchain) when staged for the
# task. 2) Online: test_task.sh, its network bootstrap retried when it failed before pytest started (no ctrf report, no
# session header). Tests that started never run twice.
VERIFIER_WRAPPER = """#!/bin/bash
# eval_q35 (worker.py) verifier wrapper: offline toolchain first (test_offline.sh, when staged), else the task's own
# test script (test_task.sh) online with its network bootstrap retried while pytest has not started; tests that
# started never run twice. Online retries stop at __BUDGET__ s, half the task's verifier timeout.
command -v tee >/dev/null 2>&1 && command -v grep >/dev/null 2>&1 || exec bash /tests/test_task.sh
log=$(mktemp 2>/dev/null || echo /logs/verifier/.eval_q35_try.log)
started() { [ -f /logs/verifier/ctrf.json ] || grep -q "test session starts" "$log"; }
if [ -f /tests/test_offline.sh ]; then
  echo "[eval_q35 verifier] offline toolchain"
  bash /tests/test_offline.sh 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if started; then rm -f "$log"; exit $rc; fi
  echo "[eval_q35 verifier] offline: pytest did not start; running the task's own script online"
fi
start=$SECONDS; n=0
while true; do
  bash /tests/test_task.sh 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if started; then rm -f "$log"; exit $rc; fi
  n=$((n + 1)); wait_s=$((15 * 2 ** n))
  if [ $n -ge 5 ] || [ $((SECONDS - start + wait_s)) -gt __BUDGET__ ]; then
    echo "[eval_q35 verifier] pytest did not start after $n tries; giving up"; rm -f "$log"; exit $rc
  fi
  echo "[eval_q35 verifier] pytest did not start (try $n); retrying in ${wait_s}s"
  sleep $wait_s
done
"""


def wrap_verifier(task):
    """Install VERIFIER_WRAPPER as the staged task's tests/test.sh (idempotent)."""
    tests = Path(task) / 'tests'
    if not (tests / 'test.sh').exists() or (tests / 'test_task.sh').exists():
        return
    try:
        timeout = float((tomllib.loads((Path(task) / 'task.toml').read_text()).get('verifier') or {}).get('timeout_sec') or 900)
    except (OSError, ValueError):
        timeout = 900.0
    (tests / 'test.sh').rename(tests / 'test_task.sh')
    (tests / 'test.sh').write_text(VERIFIER_WRAPPER.replace('__BUDGET__', str(int(min(600, timeout / 2)))))
    (tests / 'test.sh').chmod(0o755)


def _explicit_external_request(call, *, default_pip=False):
    """Conservative evidence for a tool's explicit public HTTP request, never DNS/egress."""
    import ipaddress
    from urllib.parse import urlsplit
    if not isinstance(call, dict) or call.get('name') != 'bash':
        return False
    arguments = call.get('arguments')
    if not isinstance(arguments, dict):
        return False
    command = arguments.get('command')
    if not isinstance(command, str):
        return False
    import ast
    def version_read(match):
        try:
            body = ast.parse(match[3]).body
        except (SyntaxError, ValueError):
            return match[0]
        if len(body) != 2 or not isinstance(body[0], ast.Import) or len(body[0].names) != 1:
            return match[0]
        name = body[0].names[0]
        if name.asname or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name.name):
            return match[0]
        call = body[1].value if isinstance(body[1], ast.Expr) else None
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != 'print'
                or len(call.args) != 1 or call.keywords or not isinstance(call.args[0], ast.Attribute)
                or call.args[0].attr != '__version__' or not isinstance(call.args[0].value, ast.Name)
                or call.args[0].value.id != name.name):
            return match[0]
        # This is passive version introspection only. The entire pip compound
        # still needs immutable effective-source review; it is never auto-clean.
        return match[1] + ' true'
    if re.search(r'(?:^|[;&|\n])\s*(?:timeout\s+[0-9.smhd]+\s+)?(?:/usr/(?:local/)?bin/)?(?:pip3?|python3?\s+-m\s+pip)\s', command):
        command = re.sub(r'(^|[;&|\n])\s*(?:/usr/(?:local/)?bin/)?python3?\s+-c\s+([\'"])(.*?)\2(?=\s*(?:[;&|\n]|[012]?>|$))',
                         version_read, command, flags=re.S)
    # A literal status echo does not change a request target. Other shell
    # expansion (including command substitution) remains ambiguous.
    parts = re.split(r'([;&|\n])', command)
    for index in range(0, len(parts), 2):
        if re.match(r'^\s*(?:echo|printf)\s', parts[index]):
            parts[index] = re.sub(r'\$\?|\$\{PIPESTATUS\[[0-9]+\]\}', '0', parts[index])
    command = ''.join(parts)
    # Only recognizable curl/wget or bounded pip install invocations. Dynamic/encoded targets, other Python
    # programs, transport overrides and mixed/private requests fail closed.
    if not re.search(r'(?:^|[;&|\n])\s*(?:timeout\s+[0-9.smhd]+\s+)?'
                     r'(?:env(?:\s+-u\s+[A-Za-z_]+)+\s+|DEBIAN_FRONTEND=noninteractive\s+)?'
                     r'(?:/usr/(?:local/)?bin/)?(?:curl|wget|pip3?|python3?\s+-m\s+pip|git\s+ls-remote|apt-get)\s', command):
        return False
    if re.search(r'--(?:resolve|connect-to|unix-socket|abstract-unix-socket|interface|proxy)\b|(?:^|\s)-x(?:\s|[^-])|/dev/(?:tcp|udp)/|\$(?!\?)|`|[<>]\(', command):
        return False
    import shlex
    simple = re.sub(r'\b[012]?>&[012]\b', '', command)
    simple = re.sub(r'(?:^|\s)[012]?>\s*/dev/null(?=\s|[;&|]|$)', ' ', simple)
    dns_hosts = []
    default_indexes = False
    for segment in re.split(r'[;&|\n]', simple):
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        if not words:
            continue
        if words[0] == 'timeout' and len(words) >= 3:
            words = words[2:]
        if words[0] == 'env':
            words = words[1:]
            allowed = {'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'}
            while len(words) >= 2 and words[0] == '-u' and words[1] in allowed:
                words = words[2:]
            if not words or Path(words[0]).name not in ('curl', 'wget'):
                return False
        if words[0] == 'DEBIAN_FRONTEND=noninteractive' and len(words) > 1 and words[1] == 'apt-get':
            words = words[1:]
        executable = Path(words[0]).name
        if executable in ('python', 'python3'):
            if words[1:3] != ['-m', 'pip']:
                return False
            words = ['pip', *words[3:]]
            executable = 'pip'
        if executable in ('pip', 'pip3'):
            # Package names/specifiers only. Unknown options, local/editable/
            # requirements/VCS inputs and configuration/transport overrides fail closed.
            args = list(words[1:])
            globals_ = {'--no-input', '--disable-pip-version-check', '--no-cache-dir', '-q', '--quiet'}
            while args and args[0] in globals_:
                args.pop(0)
            if not args or args.pop(0) != 'install':
                return False
            switches = globals_ | {'-U', '--upgrade', '--user', '--pre', '--prefer-binary',
                                   '--no-deps', '--force-reinstall', '--ignore-installed',
                                   '--break-system-packages', '--no-warn-script-location', '--dry-run'}
            names, explicit_index = [], False
            index = 0
            while index < len(args):
                word = args[index]
                option, equals, value = word.partition('=')
                if option in ('-i', '--index-url'):
                    if explicit_index:
                        return False
                    if not equals:
                        index += 1
                        if index >= len(args):
                            return False
                        value = args[index]
                    if not re.match(r'https?://', value):
                        return False
                    explicit_index = True
                elif word in switches:
                    pass
                elif option in ('--timeout', '--default-timeout', '--retries'):
                    if not equals:
                        index += 1
                        if index >= len(args):
                            return False
                        value = args[index]
                    try:
                        number = float(value)
                    except ValueError:
                        return False
                    if option == '--retries':
                        if not number.is_integer() or not 0 <= number <= 100:
                            return False
                    elif not 0 < number <= 3600:
                        return False
                elif re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?'
                                  r'(?:(?:==|!=|~=|>=|<=|>|<)[A-Za-z0-9.*+!_-]+'
                                  r'(?:,(?:==|!=|~=|>=|<=|>|<)[A-Za-z0-9.*+!_-]+)*)?', word):
                    names.append(word)
                else:
                    return False
                index += 1
            if not names:
                return False
            default_indexes |= not explicit_index
        elif executable == 'git':
            args = words[1:]
            if not args or args.pop(0) != 'ls-remote':
                return False
            while args and args[0] in ('--heads', '--tags', '--refs'):
                args.pop(0)
            if not args or not re.match(r'https?://', args.pop(0)):
                return False
            if any(not re.fullmatch(r'(?:HEAD|refs/[A-Za-z0-9_./*+-]+)', arg) for arg in args):
                return False
        elif executable == 'apt-get':
            args = [word for word in words[1:] if word not in ('-y', '-q', '-qq', '--yes', '--no-install-recommends', '--download-only')]
            if not args or args[0] not in ('install', 'download', 'update'):
                return False
            verb, names = args[0], args[1:]
            if (verb == 'update' and names) or (verb != 'update' and not names):
                return False
            if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9+_.:-]*(?:=[A-Za-z0-9_.:+~-]+)?', name) for name in names):
                return False
            # Like default pip, sources require a separate immutable source audit.
            default_indexes = True
        elif executable in ('curl', 'wget'):
            values = {'-m': 'number', '--max-time': 'number', '--connect-timeout': 'number',
                      '-o': 'output', '--output': 'output', '-w': 'output', '--write-out': 'output',
                      '--url': 'url'} if executable == 'curl' else {
                      '-T': 'number', '--timeout': 'number', '--connect-timeout': 'number',
                      '--read-timeout': 'number', '-O': 'output', '--output-document': 'output'}
            switches = {'--head', '--silent', '--show-error', '--fail', '--fail-with-body', '--location',
                        '--insecure', '--compressed', '--no-progress-meter', '--remote-name'} if executable == 'curl' else {
                        '--quiet', '--server-response', '--spider', '--no-check-certificate'}
            shorts = 'sSIfLvkO' if executable == 'curl' else 'qS'
            targets = []
            index = 1
            while index < len(words):
                word = words[index]
                option, equals, attached = word.partition('=')
                if option not in values and word.startswith('-') and not word.startswith('--') and word[:2] in values:
                    option, equals, attached = word[:2], '=', word[2:]
                if option in values:
                    if not equals:
                        index += 1
                        if index >= len(words):
                            return False
                        attached = words[index]
                    if not attached:
                        return False
                    kind = values[option]
                    if kind == 'url':
                        targets.append(attached)
                    elif kind == 'number':
                        try:
                            value = float(attached)
                        except ValueError:
                            return False
                        if not 0 < value < float('inf'):
                            return False
                elif word.startswith('-'):
                    if word not in switches and not (len(word) > 1 and not word.startswith('--') and all(c in shorts for c in word[1:])):
                        return False
                else:
                    targets.append(word)
                index += 1
            if not targets or any(not re.match(r'https?://', w) for w in targets):
                return False
        elif executable in ('head', 'tail'):
            index = 1
            while index < len(words):
                if words[index] in ('-n', '-c'):
                    index += 1
                if index >= len(words) or not re.fullmatch(r'-?[0-9]+', words[index]):
                    return False
                index += 1
        elif executable in ('echo', 'printf'):
            continue
        elif executable == 'cd' and len(words) == 2 and words[1].startswith('/'):
            continue
        elif executable in ('true', 'false') and len(words) == 1:
            continue
        elif executable == 'command' and len(words) > 2 and words[1] in ('-v', '-V'):
            if any(not re.fullmatch(r'[A-Za-z0-9_./+-]+', word) or word.startswith('-') for word in words[2:]):
                return False
        elif executable == 'which' and len(words) > 1:
            if any(not re.fullmatch(r'[A-Za-z0-9_./+-]+', word) or word.startswith('-') for word in words[1:]):
                return False
        elif executable == 'getent' and len(words) > 2 and words[1] == 'hosts':
            dns_hosts.extend(words[2:])
        elif executable == 'cat':
            static = {'/etc/hosts', '/etc/resolv.conf', '/etc/apt/sources.list',
                      '/opt/aa-runtime-cache/public-packages.json',
                      '/opt/aa-runtime-cache/apt/apt.conf', '/opt/aa-runtime-cache/apt/sources.list'}
            if len(words) < 2 or any(word not in static for word in words[1:]):
                return False
        elif executable == 'find':
            # Only passive metadata predicates; never read a filename list or
            # execute commands/write output files while suppressing a result.
            flags = {'-H', '-L', '-P', '-maxdepth', '-mindepth', '-type', '-name', '-iname',
                     '-path', '-ipath', '-print', '-print0', '-ls', '-empty', '-readable',
                     '-writable', '-executable', '-o', '-or', '-a', '-and', '-not'}
            if any(word.startswith('-') and word not in flags for word in words[1:]):
                return False
        elif executable in ('ls', 'pwd'):
            continue
        else:
            return False
    urls = re.findall(r'https?://[^\s\'"<>`\\;|&(){}]+', command)
    if default_indexes and not default_pip:
        return False
    if not urls and not default_indexes:
        return False
    if re.search(r'\b(?!https?://)[A-Za-z][A-Za-z0-9+.-]*://', command):
        return False
    hosts = list(dns_hosts)
    try:
        for url in urls:
            parsed = urlsplit(url)
            _ = parsed.port  # malformed/out-of-range ports are not unambiguous evidence
            if parsed.username or parsed.password or not parsed.hostname:
                return False
            hosts.append(parsed.hostname.lower().rstrip('.'))
        # Private numeric targets anywhere in a mixed shell command are ambiguous.
        for ip in re.findall(r'(?<![\w.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\w.])', command):
            if not ipaddress.ip_address(ip).is_global:
                return False
        for host in hosts:
            try:
                if not ipaddress.ip_address(host).is_global:
                    return False
            except ValueError:
                if (not re.fullmatch(r'[a-z0-9.-]+', host) or re.fullmatch(r'[0-9.]+', host) or '.' not in host
                        or host.endswith(('.localhost', '.local', '.internal', '.lan', '.test', '.invalid', '.cluster.internal'))):
                    return False
    except ValueError:
        return False
    return True


def _explicit_public_python_request(call):
    """Only a literal requests.get diagnostic, not arbitrary Python or model APIs."""
    import ast
    if not isinstance(call, dict) or call.get('name') != 'bash' or not isinstance(call.get('arguments'), dict):
        return False
    command = call['arguments'].get('command')
    if not isinstance(command, str):
        return False
    match = re.fullmatch(r'\s*(?:timeout\s+[0-9.smhd]+\s+)?python3?\s+-\s*<<[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]\n(.*?)\n\1\s*', command, re.S)
    if not match:
        return False
    try:
        tree = ast.parse(match[2])
    except (SyntaxError, ValueError):
        return False
    allowed = (ast.Module, ast.Import, ast.alias, ast.For, ast.Name, ast.Load, ast.Store,
               ast.List, ast.Tuple, ast.Constant, ast.Try, ast.ExceptHandler, ast.Assign,
               ast.Expr, ast.Call, ast.Attribute, ast.Subscript, ast.Slice, ast.keyword)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        return False
    urls, variables = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name != 'requests' or alias.asname for alias in node.names):
                return False
        elif isinstance(node, ast.For):
            if (not isinstance(node.target, ast.Name) or node.target.id in ('requests', 'print', 'len', 'str', 'type')
                    or not isinstance(node.iter, (ast.List, ast.Tuple)) or node.orelse):
                return False
            values = [value.value for value in node.iter.elts if isinstance(value, ast.Constant) and isinstance(value.value, str)]
            if len(values) != len(node.iter.elts) or node.target.id in variables:
                return False
            variables[node.target.id] = values
        elif isinstance(node, ast.Assign):
            if (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
                    or node.targets[0].id in ('requests', 'print', 'len', 'str', 'type')
                    or not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Attribute)
                    or not isinstance(node.value.func.value, ast.Name) or node.value.func.value.id != 'requests'
                    or node.value.func.attr != 'get'):
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ('print', 'len', 'str', 'type'):
                continue
            if (not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name)
                    or node.func.value.id != 'requests' or node.func.attr != 'get' or len(node.args) != 1):
                return False
            argument = node.args[0]
            values = [argument.value] if isinstance(argument, ast.Constant) and isinstance(argument.value, str) else variables.get(getattr(argument, 'id', ''), [])
            if not values or len(node.keywords) != 1 or node.keywords[0].arg != 'timeout':
                return False
            timeout = node.keywords[0].value
            if not isinstance(timeout, ast.Constant) or not isinstance(timeout.value, (int, float)) or not 0 < timeout.value <= 60:
                return False
            urls.extend(values)
    return bool(urls) and _explicit_external_request({'name': 'bash', 'arguments': {'command': 'curl ' + ' '.join(urls)}})


def _unresolved_agent_host_probes(call, text, bridge_url, bridge_config_sha256):
    """Known host bridge/proxy probes are unresolved, never a public-target approval."""
    import ipaddress
    import shlex
    from urllib.parse import urlsplit
    if not isinstance(bridge_url, str) or not re.fullmatch(r'[0-9a-f]{64}', str(bridge_config_sha256)):
        return None
    try:
        bridge = urlsplit(bridge_url)
        address = ipaddress.ip_address(bridge.hostname or '')
        if (bridge.scheme != 'http' or bridge.username or bridge.password or bridge.path != '/v1'
                or not address.is_private or address.is_loopback or not bridge.port):
            return None
    except ValueError:
        return None
    if not isinstance(call, dict) or call.get('name') != 'bash' or not isinstance(call.get('arguments'), dict):
        return None
    command = call['arguments'].get('command')
    if not isinstance(command, str) or re.search(r'\$|`|[<>]\(', command):
        return None
    command = re.sub(r'\b[012]?>&[012]\b', '', command)
    targets = []
    for segment in re.split(r'[;&|\n]', command):
        try:
            words = shlex.split(segment)
        except ValueError:
            return None
        if not words:
            continue
        if words[0] == 'timeout':
            if len(words) < 3 or not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', words[1]) or not 0 < float(words[1]) <= 60:
                return None
            words = words[2:]
        if words in (['set', '-x'], ['true']):
            continue
        if words[0] in ('head', 'tail') and len(words) == 2 and re.fullmatch(r'-[0-9]+', words[1]):
            continue
        if words[:2] == ['bash', '-c'] and len(words) == 3:
            match = re.fullmatch(r'echo\s*>\s*/dev/tcp/([0-9.]+)/([0-9]+)', words[2])
            if not match:
                return None
            target = f'tcp://{match[1]}:{match[2]}'
        elif words[0] == 'curl':
            args, urls, bypass = words[1:], [], False
            while args:
                option = args.pop(0)
                if option == '--noproxy':
                    if not args or args.pop(0) != '*' or bypass:
                        return None
                    bypass = True
                elif re.fullmatch(r'-[sSvI]+', option):
                    continue
                elif re.match(r'https?://', option):
                    urls.append(option)
                else:
                    return None
            if len(urls) != 1:
                return None
            target = urls[0]
            if bypass and target != f'http://{bridge.hostname}:{bridge.port}/v1/models':
                return None
        else:
            return None
        try:
            parsed = urlsplit(target)
            hostport = (parsed.hostname, parsed.port)
        except ValueError:
            return None
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            return None
        if hostport == (bridge.hostname, bridge.port):
            if (parsed.scheme == 'tcp' and parsed.path) or (parsed.scheme == 'http' and parsed.path != '/v1/models') or parsed.scheme not in ('tcp', 'http'):
                return None
        elif hostport == ('10.22.0.107', 8080):
            if parsed.scheme != 'http' or parsed.path not in ('', '/'):
                return None
        elif not _explicit_external_request({'name': 'bash', 'arguments': {'command': 'curl ' + target}}):
            return None
        targets.append(target)
    if (not any(urlsplit(target).hostname == bridge.hostname for target in targets)
            or not any(urlsplit(target).hostname == '10.22.0.107' for target in targets)):
        return None
    for host in (bridge.hostname, '10.22.0.107'):
        if not re.search(re.escape(host) + r'[^\n]{0,180}Network is unreachable', text):
            return None
    return {'reason': 'agent_host_egress_unresolved', 'probe_targets': sorted(set(targets)),
            'bridge_url': bridge_url, 'bridge_config_sha256': bridge_config_sha256,
            'known_proxy_url': 'http://10.22.0.107:8080'}


def _continued_agent_model_response(lines, after):
    """Bind a later paid primary-model response, excluding synthetic context stops."""
    import hashlib
    for index in range(after + 1, len(lines)):
        try:
            event = json.loads(lines[index])
        except ValueError:
            continue
        msg = event.get('message') if isinstance(event, dict) else None
        if not isinstance(msg, dict):
            continue
        usage = msg.get('usage') or {}
        output = usage.get('output') if isinstance(usage, dict) else None
        response = msg.get('responseId')
        if (event.get('type') == 'message' and msg.get('role') == 'assistant'
                and msg.get('api') == 'openai-completions' and msg.get('provider') == 'harbor-endpoint'
                and isinstance(msg.get('content'), list) and msg['content']
                and msg.get('stopReason') != 'error' and not msg.get('errorMessage')
                and isinstance(response, str) and response.startswith('chatcmpl-')
                and type(output) is int and output > 0):
            return {'line_index': index, 'response_id': response,
                    'event_sha256': hashlib.sha256(lines[index].encode()).hexdigest()}
    return None


def _pi_session_lines(path, *, errors='replace'):
    """Read physical JSONL records; hashes/indices agree with trace validation."""
    text = path.read_bytes().decode('utf-8', errors=errors)
    if not text:
        return []
    return [line.removesuffix('\r') for line in text.removesuffix('\n').split('\n')]


def _offline_agent_proxy_failure(path, *, review=None, bridge_url=None, bridge_config_sha256=None):
    """Keep backend/private hits; default pip policy denials require source-index review."""
    import hashlib
    calls = {}
    lines = _pi_session_lines(path)
    for line_number, line in enumerate(lines):
        try:
            event = json.loads(line)
        except ValueError:
            event = {}
        if not isinstance(event, dict):
            event = {}
        message = event.get('message')
        if not isinstance(message, dict):
            message = {}
        content = message.get('content')
        if not isinstance(content, list):
            content = []
        if event.get('type') == 'message' and message.get('role') == 'assistant':
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'toolCall':
                    if isinstance(block.get('id'), str):
                        calls[block['id']] = block
        call_id = message.get('toolCallId')
        call = calls.get(call_id, {}) if isinstance(call_id, str) else {}
        linked_bash = (event.get('type') == 'message' and message.get('role') == 'toolResult'
                       and message.get('toolName') == 'bash')
        arguments = call.get('arguments')
        command = arguments.get('command', '') if isinstance(arguments, dict) else ''
        # APT's ENETUNREACH wording is not in the legacy proxy regex. Retain it
        # as infrastructure unless this exact tool result earns an apt review
        # marker; private/mixed/unknown sources must not silently become clean.
        apt_text = '\n'.join(block['text'] for block in content
                             if isinstance(block, dict) and isinstance(block.get('text'), str)) if linked_bash else ''
        apt_denial = re.search(r'Cannot initiate the connection to (?:172\.22\.0\.107|proxy):8080[^\n]*Network is unreachable', apt_text)
        if linked_bash and review is not None:
            texts = [block.get('text') if isinstance(block, dict) else None for block in content]
            text = '\n'.join(texts) if all(isinstance(value, str) for value in texts) else ''
            unresolved = _unresolved_agent_host_probes(call, text, bridge_url, bridge_config_sha256)
            continued = False
            if unresolved:
                for later in lines[line_number + 1:]:
                    try:
                        later = json.loads(later)
                    except ValueError:
                        continue
                    msg = later.get('message') if isinstance(later, dict) else None
                    if (isinstance(later, dict) and later.get('type') == 'message' and isinstance(msg, dict) and msg.get('role') == 'assistant' and isinstance(msg.get('content'), list)
                            and msg['content'] and msg.get('stopReason') != 'error' and not msg.get('errorMessage')):
                        continued = True
                        break
            if unresolved and continued and not message.get('errorMessage') and message.get('stopReason') != 'error':
                review.append({**unresolved, 'tool_call_id': call_id, 'command': command,
                               'tool_result_sha256': hashlib.sha256(line.encode()).hexdigest()})
                continue
        package = linked_bash and isinstance(command, str) and bool(re.search(
            r'(?:^|[;&|\n])\s*(?:timeout\s+[0-9.smhd]+\s+)?'
            r'(?:/usr/(?:local/)?bin/)?(?:pip3?|python3?\s+-m\s+pip)\s', command))
        source_kinds = {'pip'} if package else set()
        if linked_bash and isinstance(command, str):
            if re.search(r'(?:^|[;&|\n])\s*(?:timeout\s+[0-9.smhd]+\s+)?git\s+ls-remote\s', command):
                source_kinds.add('git')
            if re.search(r'(?:^|[;&|\n])\s*(?:timeout\s+[0-9.smhd]+\s+)?(?:DEBIAN_FRONTEND=noninteractive\s+)?apt-get\s', command):
                source_kinds.add('apt')
        # One result cannot prove which of several inherited manager configs failed.
        source = next(iter(source_kinds)) if len(source_kinds) == 1 else None
        python_public = linked_bash and _explicit_public_python_request(call)
        public = linked_bash and not source_kinds and (_explicit_external_request(call) or python_public)
        package = package and _explicit_external_request(call, default_pip=True)
        source = source if ((source == 'pip' and package) or (source in ('git', 'apt') and _explicit_external_request(call, default_pip=True))) else None
        if source or python_public:
            # Even explicit -i does not clear inherited extra-index-url config.
            # Only actual pip proxy-denial text enters the source-review state.
            texts = [block.get('text', '') if isinstance(block, dict) else None for block in content]
            valid_text = all(isinstance(text, str) for text in texts)
            text = '\n'.join(texts) if valid_text else ''
            if source == 'pip':
                source = source if valid_text and re.search(r"ProxyError\(['\"]Cannot connect to proxy", text) else None
            elif source == 'git':
                source = source if valid_text and 'fatal: unable to access' in text and PROXY_FAIL.search(text) else None
            elif source == 'apt':
                source = source if valid_text and re.search(r'Cannot initiate the connection to (?:172\.22\.0\.107|proxy):8080[^\n]*Network is unreachable', text) else None
            if not valid_text:
                public = False
            for indexes in re.findall(r'Looking in (?:indexes|links): ([^\n]+)', text):
                urls = re.findall(r'https?://[^\s,]+', indexes)
                if not urls or not _explicit_external_request({'name': 'bash', 'arguments': {
                        'command': 'curl ' + ' '.join(urls)}}):
                    source = None
            if source in ('apt', 'git'):
                urls = re.findall(r'https?://[^\s\'"<>`]+', text)
                if any(not _explicit_external_request({'name': 'bash', 'arguments': {'command': 'curl ' + url}}) for url in urls):
                    source = None
        if (linked_bash and review is not None and not public and not source
                and isinstance(command, str) and command and call.get('name') == 'bash'
                and isinstance(call_id, str) and content
                and all(isinstance(block, dict) and block.get('type') == 'text'
                        and isinstance(block.get('text'), str) for block in content)
                and (apt_denial or PROXY_FAIL.search(apt_text))
                and not message.get('errorMessage') and message.get('stopReason') != 'error'
                and isinstance(bridge_url, str) and bridge_url
                and isinstance(bridge_config_sha256, str)
                and re.fullmatch(r'[0-9a-f]{64}', bridge_config_sha256)):
            continuation = _continued_agent_model_response(lines, line_number)
            if continuation:
                review.append({'reason': 'agent_tool_network_origin_not_proven',
                               'tool_call_id': call_id, 'command': command,
                               'tool_result_sha256': hashlib.sha256(line.encode()).hexdigest(),
                               'bridge_url': bridge_url, 'bridge_config_sha256': bridge_config_sha256,
                               'continuation': continuation})
                continue
        if apt_denial and source != 'apt':
            return apt_denial
        if public or (source and review is not None):
            proven = (PROXY_FAIL.search(line) or source == 'apt' or
                      (python_public and re.search(r'ProxyError.*Unable to connect to proxy', text)))
            if proven:
                if source:
                    reason = {'pip': 'pip_effective_indexes_not_proven', 'git': 'git_effective_urls_not_proven',
                              'apt': 'apt_effective_sources_not_proven'}[source]
                    review.append({'reason': reason, 'tool_call_id': call_id,
                                   'command': call['arguments']['command'],
                                   'tool_result_sha256': hashlib.sha256(line.encode()).hexdigest()})
            continue
        if apt_denial:
            return apt_denial
        evidence = line
        if (event.get('type') == 'message' and message.get('role') == 'assistant'
                and isinstance(message.get('content'), list) and message.get('stopReason') != 'error'
                and not message.get('errorMessage')):
            # Assistant prose and source code are not observations of a transport
            # failure. Retain every explicit structured error field.
            evidence = json.dumps({**event, 'message': {**message, 'content': []}})
        match = PROXY_FAIL.search(evidence)
        if match:
            return match
    return None


def proxy_failure(trial, *, agent_internet=True, offline_review=None):
    """Cluster-proxy failure in agent evidence or a verifier whose tests never started.

    Explicit offline policy excludes only a linked toolResult for a recognizable
    external curl/wget request. Recognizable pip installs, including public -i,
    stop retries only with a mandatory effective-index source-review marker. Other linked tool
    denials require a later paid primary response and bound trial config to enter mandatory manual
    origin review. Structured model errors and verifier failures retain infrastructure handling.
    """
    trial = locate(trial)   # a packed attempt (raw.zip) too
    sources = [('agent', p) for p in pi_sessions(trial)]
    if not (trial / 'verifier' / 'ctrf.json').exists():
        sources.append(('verifier', trial / 'verifier' / 'test-stdout.txt'))
    for where, path in sources:
        try:
            if where == 'agent' and agent_internet is False:
                config_path = trial / 'config.json'
                bridge_url, bridge_sha = None, None
                try:
                    import hashlib
                    config_bytes = config_path.read_bytes()
                    config = json.loads(config_bytes)
                    bridge_url = config['agent']['env']['OPENAI_BASE_URL']
                    bridge_sha = hashlib.sha256(config_bytes).hexdigest()
                except (OSError, ValueError, KeyError, TypeError):
                    pass
                match = _offline_agent_proxy_failure(path, review=offline_review, bridge_url=bridge_url,
                                                    bridge_config_sha256=bridge_sha)
            else:
                match = PROXY_FAIL.search(path.read_text(errors='replace'))
        except OSError:
            continue
        if match:
            return f'{where}: {match.group(0)[:200]}'
    return None


# Last verifier output line of a test.sh network bootstrap (apt, the uv installer, uv/pip package downloads) or of the
# wrapper retrying it. On 2026-10-29 ~10:20-11:10 UTC the cluster network was slow, not refused: bootstraps outran the
# tasks' verifier timeouts (60-900 s) before pytest started, and the attempts scored 0 (tbmix runs: 3-9 each).
VERIFIER_BOOTSTRAP = re.compile(
    r'downloading uv|^Get:\d|Unpacking |Setting up |Selecting previously unselected|Preparing to unpack|'
    r'Reading (package lists|state information)|Building dependency tree|Processing triggers|^\s*Downloading |'
    r'(Resolved|Prepared|Installed|Audited) \d+ packages?|Creating virtual environment|Using (CPython|Python) |'
    r'^Collecting |installing to |everything.s installed|no checksums to verify|^\s*uvx?$|source \$HOME|'
    r'\[eval_q35 verifier\] pytest did not start')


def verifier_bootstrap_timeout(trial, exception_type):
    """A verifier timeout inside the tests' network bootstrap: pytest never started (no ctrf report, no session header)
    and the verifier's last output line is bootstrap activity, so the tests never judged the agent's work. Tasks
    whose verifier runs no bootstrap (e.g. a bare `python ref_eval.py`) never match. Returns 'verifier: ...' or None."""
    if exception_type != 'VerifierTimeoutError':
        return None
    v = locate(trial) / 'verifier'
    if (v / 'ctrf.json').exists():
        return None
    try:
        lines = [x for x in (v / 'test-stdout.txt').read_text(errors='replace').splitlines() if x.strip()]
    except OSError:
        return None
    if not lines or any('test session starts' in x for x in lines):
        return None
    return f'verifier: timed out in its bootstrap: {lines[-1].strip()[:160]}' if VERIFIER_BOOTSTRAP.search(lines[-1]) else None


def log(msg):
    print(f'{now()} {msg}', flush=True)


def pi_sessions(trial):
    return sorted(locate(trial).glob('agent/pi/sessions/**/*.jsonl'))


def last_assistant(trial):
    last = None
    for path in pi_sessions(trial):
        for line in _pi_session_lines(path):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            message = event.get('message')
            if not isinstance(message, dict):
                continue
            if event.get('type') == 'message' and message.get('role') == 'assistant':
                last = message
    return last


def _agent_startup_configuration_failure(trial, result):
    """Causal harness configuration traceback, not a model-issued chmod or missing-log inference."""
    exc = result.get('exception_info') or {}
    if not isinstance(exc, dict) or exc.get('exception_type') != 'NonZeroAgentExitCodeError':
        return False
    message, trace = exc.get('exception_message'), exc.get('exception_traceback')
    if not isinstance(message, str) or not isinstance(trace, str):
        return False
    if (not message.startswith('Command failed (exit 1): chmod 600 /tmp/harbor-pi-agent/models.json\n')
            or not re.search(r'chmod: changing permissions of [\'"]/tmp/harbor-pi-agent/models.json[\'"]: (?:Operation not permitted|Permission denied)', message)):
        return False
    if (not re.search(r'harbor/agents/installed/pi\.py[^\n]*in run\n\s+await self\._write_custom_models_json\(', trace)
            or 'in _write_custom_models_json' not in trace or message.strip() not in trace):
        return False
    agent = result.get('agent_result') or {}
    if not isinstance(agent, dict) or any(agent.get(k) not in (None, 0) for k in ('n_input_tokens', 'n_output_tokens')):
        return False
    for path in pi_sessions(trial):
        try:
            for line in _pi_session_lines(path, errors='strict'):
                event = json.loads(line)
                if not isinstance(event, dict):
                    return False
                msg = event.get('message') or {}
                if not isinstance(msg, dict) or msg.get('role') in ('assistant', 'toolResult'):
                    return False
        except (OSError, ValueError):
            return False
    wire = locate(trial).parent / 'pi-cont16k' / 'wire-requests.jsonl'
    try:
        if wire.exists() and wire.read_text().strip():
            return False
    except OSError:
        return False
    return True


def _grading_pending(trial):
    """A typed evaluator adapter limitation resolves Gen, but cannot receive a score."""
    path = locate(trial) / 'verifier' / 'grading-pending.json'
    try:
        import hashlib
        raw = path.read_bytes()
        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError('duplicate pending-grading field')
                result[name] = value
            return result
        marker = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(marker, dict):
            return None
        evidence = marker.get('evidence')
        if (type(marker.get('schema_version')) is not int or marker['schema_version'] != 1
                or marker.get('kind') != 'caffe_data_adapter_unsupported'
                or marker.get('task') != 'caffe-cifar-10' or marker.get('grading_pending') is not True
                or marker.get('regenerate') is not False or not isinstance(marker.get('reason'), str)
                or not marker['reason'].strip() or not isinstance(evidence, dict)):
            return None
        hashes = ('submitted_network_sha256', 'official_test_sha256', 'descriptor_sha256', 'grader_sha256')
        if any(not isinstance(evidence.get(k), str) or not re.fullmatch(r'[0-9a-f]{64}', evidence[k]) for k in hashes):
            return None
        layers = evidence.get('input_layers')
        if (not isinstance(layers, list) or any(not isinstance(layer, dict)
                or not all(isinstance(layer.get(k), str) and layer[k] for k in ('name', 'type'))
                or not isinstance(layer.get('configuration_sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', layer['configuration_sha256']) for layer in layers)):
            return None
        return {'grading_pending': True, 'grading_review_required': True,
                'grading_pending_evidence': marker, 'grading_pending_sha256': hashlib.sha256(raw).hexdigest(),
                'score': None, 'infrastructure_failure': False}
    except (OSError, ValueError, TypeError):
        return None


def classify(raw, returncode, stopped, harness='pi', *, agent_internet=True):
    """Outcome of one physical attempt from its Harbor trial directory. stopped: False, True (worker stopped) or
    'endpoint_moved' (the run's serving job moved and the attempt's endpoint is gone)."""
    trials = [p.parent for p in sorted(Path(raw).glob('*/*/result.json')) if p.parent.name != 'pi-cont16k']
    out = {'returncode': returncode, 'trial_dir': str(trials[0]) if trials else None}
    if stopped == 'endpoint_moved':
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'endpoint_moved',
                'score': None, 'exception_type': 'EndpointMoved'}
    if stopped:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'worker_stopped',
                'score': None, 'exception_type': 'WorkerStopped'}
    if not trials:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'no_harbor_result',
                'score': None, 'exception_type': 'NoHarborResult'}
    result = read(trials[0] / 'result.json', {})
    exc = result.get('exception_info') or {}
    etype = exc.get('exception_type') or ''
    message = exc.get('exception_message') or ''
    rewards = (result.get('verifier_result') or {}).get('rewards') or {}
    reward = rewards.get('reward')
    out.update(exception_type=etype or None, exception_message=message[:2000] or None,
               reward=reward, score=float(reward) if isinstance(reward, (int, float)) else None)
    if _agent_startup_configuration_failure(trials[0], result):
        return {**out, 'score': None, 'infrastructure_failure': True,
                'infrastructure_reason': 'agent_startup_configuration'}
    review = []
    proxy = proxy_failure(trials[0], agent_internet=agent_internet, offline_review=review)
    if review:
        out.update(offline_proxy_review_required=True, offline_proxy_review_evidence=review)
    if proxy:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'proxy_outage', 'proxy_failure': proxy}
    stall = verifier_bootstrap_timeout(trials[0], etype)
    if stall:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'verifier_bootstrap_timeout',
                'verifier_stall': stall}
    if sandbox_setup_failure(result):
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': 'userns_setup'}
    if harness == 'terminus-2':
        clock = read(trials[0] / 'agent' / 'env_clock.json', {})
        out.update(end_state=clock.get('state'), output_cap_hits=clock.get('output_cap_hits'),
                   requests=clock.get('requests'))
        infra = None
        if reward is None and any(t in etype for t in INFRA):
            infra = etype
        elif reward is None and (etype in LLM_INFRA or clock.get('error') in LLM_INFRA):
            infra = 'llm_backend_unreachable'
        if infra:
            return {**out, 'infrastructure_failure': True, 'infrastructure_reason': infra}
        pending = _grading_pending(trials[0]) if reward is None else None
        if pending:
            return {**out, **pending}
        if out['score'] is None:
            out['score'] = 0.0
            out['unscored'] = True
        out['infrastructure_failure'] = False
        return out
    has_session = bool(pi_sessions(trials[0]))
    last = last_assistant(trials[0]) if has_session else None
    error = str((last or {}).get('errorMessage', ''))
    infra = None
    if not has_session and etype == 'NonZeroAgentExitCodeError' and (
            ('GLIBC_' in message and 'not found' in message) or 'Unknown option:' in message):
        infra = 'pi_startup_before_inference'
    elif reward is None and any(t in etype for t in INFRA):
        infra = etype
    elif last and last.get('stopReason') == 'error' and re.search(r'response[ _]budget[ _]accounting', error, re.I):
        # the per-rollout response-budget guard could not measure a generation (tb_agents/response_budget_proxy.py,
        # e.g. /detokenize failed twice): infrastructure, never the model's answer (#531)
        infra = 'response_budget_accounting'
    elif last and last.get('stopReason') == 'error' and re.search(
            r'connection error|fetch failed|ECONNREFUSED|ECONNRESET|network error|socket hang up|'
            r'50[0-9].{0,30}(internal server error|bad gateway|service unavailable)|Server got itself in trouble|'
            r'Cannot verify rendered Pi context|Unable to verify context headroom', error, re.I):
        # Pi exhausted its HTTP retries: the model was cut off by serving, not by its own actions.
        infra = 'pi_backend_unreachable'
    elif last and last.get('stopReason') == 'error' and re.search(r'request timed out|ETIMEDOUT', error, re.I):
        # Every try of one model request outlived Pi's 30-min request timeout: a stalled or very slow server (cpu208,
        # 2026-11-01: 126 of 500 attempts), never seen on healthy serving.
        infra = 'pi_backend_timeout'
    if infra:
        return {**out, 'infrastructure_failure': True, 'infrastructure_reason': infra, 'pi_error': error[:500] or None}
    pending = _grading_pending(trials[0]) if reward is None else None
    if pending:
        return {**out, **pending}
    if out['score'] is None:
        # Model-side outcomes without a verifier reward (e.g. verifier timeout) count as failures.
        out['score'] = 0.0
        out['unscored'] = True
    out['infrastructure_failure'] = False
    if last and last.get('stopReason') == 'error':
        out['pi_error'] = error[:500]
    return out


def temperature_env(c, p, attempt):
    """#167: several sampling temperatures in one run, one block of attempts each: attempts [i * k, (i + 1) * k) use
    temperatures[i], k = attempts / len(temperatures). The proxy then samples at that temperature and the protocol's
    top_p (gen/pi_cont16k.py -> sticky_proxy --temperature/--top-p). A run without "temperatures": nothing (the
    proxy's protocol defaults, 1.0 / 0.95)."""
    temps = c.get('temperatures')
    if not temps:
        return {}
    k = max(1, c['attempts'] // len(temps))
    return {'PI_TEMPERATURE': str(temps[min(attempt // k, len(temps) - 1)]), 'PI_TOP_P': str(p['top_p'])}


class Worker:
    # One worker for eval_q35 and aa_evals pi_eval, which subclasses it (#480). Where the two pipelines ran different
    # worker rules, these class settings pick them: eval_q35's here, aa-evals' in PiEvalWorker
    # (aa_evals/common/pi_eval/worker.py).
    DISKCAP_WITHOUT_KEY = True   # a run config without "diskcap" gets common.DISKCAP_DEFAULT (eval #372); False: off
    IDLE_EXIT_S = IDLE_EXIT_S    # nothing to claim this long: exit (eval #258: 300 s; aa-evals pi_eval too since #531)
    RESOURCE_LOG = True          # workers/<name>.resources.jsonl every minute (eval #258)
    # Pi gets the task instruction as its last argument, as on eval, which never had aa-evals' stdin launch (#203;
    # the shared harness's default since #487). Pi trims stdin, so the stdin launch drops the instruction's trailing
    # newline from the first user message of every request. False: the harness default (stdin), as AA evals run it.
    INSTRUCTION_ARGV = True

    def __init__(self, a):
        self.a = a
        self.run = Path(a.run)
        self.state = State(self.run)
        self.config = read(self.run / 'config.json')
        self.name = f"{os.environ.get('CONDOR_JOB_ID', 'local')}-{socket.gethostname()}-{os.getpid()}"
        self.running = {}
        self.tasks_dir = SCRATCH / 'tasks'
        self.proxy_state = (0.0, True)
        diskcap_config(self.diskcap_cfg)   # a malformed "diskcap" stops the worker here, not every attempt
        self.one_per_worker = diskcap_one_per_worker(self.diskcap_cfg)

    @property
    def diskcap_cfg(self):
        return run_diskcap_config(self)

    @property
    def diskcap_worker_disk_mb(self):
        # the scratch a task's cap is weighed against for one_per_worker (pi_eval: its run's worker disk)
        return DISKCAP_WORKER_DISK_MB

    def claim_tasks(self, available):
        """Tasks this worker may claim now: a one-per-worker task (common.diskcap_one_per_worker: listed in the config,
        or its effective cap above a quarter of the scratch) not while one of its attempts runs here. Other workers still
        claim it, and this one again once it ends, so it waits at most one attempt. Its seed, protocol and verifier do not
        depend on the worker; only when it starts does, and the tail rule's last-few cut pauses while such an attempt is
        held back and gives it TAIL_LAST_MIN once claimed (orchestrate.cut_tails)."""
        busy = {parse_key(k)[0] for k in self.running} & self.one_per_worker
        return available - busy if busy else available

    def stage(self):
        # Mixed benchmarks (common.BENCHMARKS 'bundles') stage every bundle into one tasks/ (task names are unique).
        bundles = [Path(b) for b in self.config.get('bundles') or [self.config.get('bundle', str(BUNDLE))]]
        if not self.tasks_dir.exists():
            tmp = SCRATCH / 'tasks.stage'
            shutil.rmtree(tmp, ignore_errors=True)
            for bundle in bundles:
                with tarfile.open(bundle) as tf:
                    tf.extractall(tmp, filter='data')
            os.rename(tmp / 'tasks', self.tasks_dir)
        for task in self.tasks_dir.iterdir():
            wrap_verifier(task)
        if not self.config.get('online_verifier'):   # config opt-out: the tasks' own (online) bootstrap only
            # the toolchain manifest keys tasks by bundle stem: each call stages its own bundle's tasks. A run config
            # `verifier_toolchain_bundle` names the manifest's bundle for a derived bundle (AA's m36 bundle-aptfix.tar
            # = tb21_6h with 10 fixed test scripts, #531 B5): only tasks whose test.sh is byte-identical to the one
            # the record was validated with go offline (offline_script's sha check); the others stay online
            alias = self.config.get('verifier_toolchain_bundle')
            if alias and len(bundles) > 1:   # one alias cannot stand for several bundles' records
                raise SystemExit('verifier_toolchain_bundle needs a single-bundle run')
            n = sum(verifier_toolchain.stage_bundle(self.tasks_dir, stem, SCRATCH / 'toolchain')
                    for stem in ([alias] if alias else [b.stem for b in bundles]))
            log(f'offline verifier toolchain staged for {n} tasks')
        return {p.name for p in self.tasks_dir.iterdir()}

    def proxy_ok(self):
        """The sandboxes' internet path works: one request through the cluster proxy, cached 60 s. While it fails the
        worker claims nothing (an outage pauses the eval instead of burning attempts)."""
        if time.time() - self.proxy_state[0] < 60:
            return self.proxy_state[1]
        proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        ok = True
        if proxy:
            ok = subprocess.run(['curl', '-sS', '-o', '/dev/null', '-m', '20', '-x', proxy, 'https://pypi.org/simple/pip/'],
                                capture_output=True).returncode == 0
        if ok != self.proxy_state[1]:
            log(f'cluster proxy {"back" if ok else "DOWN: not claiming"}')
        self.proxy_state = (time.time(), ok)
        return ok

    def env(self, k, task, attempt):
        c, p = self.config, {**PROTOCOL, **self.config.get('protocol', {})}   # the run's pinned protocol
        ctx, cap = str(p['context_window']), str(p['max_output_tokens'])
        s = seed(task, attempt, c.get('seed_namespace', SEED_NAMESPACE))
        e = {**os.environ, 'REPO': str(REPO), 'MAX_LEN': ctx, 'MAX_TOKENS': cap,
             'HARBOR_NS_LOCAL': str(SCRATCH / 'harbor_ns'), 'HARBOR_NS_SHARED': '/data/aai/tbench/harbor_ns',
             'TMPDIR': str(SCRATCH / 'tmp'), 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
             'no_proxy': 'localhost,127.0.0.1,localnet,.internal.cluster.internal',
             'NO_PROXY': 'localhost,127.0.0.1,localnet,.internal.cluster.internal'}
        if c.get('harness') == 'terminus-2':
            # Harbor's terminus-2 on this host via litellm (tb_agents/terminus_eval.py: env clock, episode end at
            # the context limit with the verifier, per-request seeds, KV gate); no summarization.
            ak = {'enable_summarize': 'false', 'proactive_summarization_threshold': 0, 'sampling_seed': s,
                  'kv_gate': c.get('kv_gate', '0.80,0.72'), 'kv_gate_file': self.run / 'serve' / 'kv.json',
                  'env_budget_multiplier': c.get('envclock_mult', 1) if c.get('envclock') else 0,
                  'tail_cut_file': self.cut_file(k)}
            extra = ' '.join(f'--ak {x}={v}' for x, v in ak.items())
            if c.get('envclock'):
                extra += f" --agent-timeout-multiplier {c['backstop_multiplier']:g}"
            memcap_env(e, c, self.run / 'attempts' / k)
            diskcap_env(e, run_diskcap_config(self), self.run / 'attempts' / k, task)
            e.update({'AGENT_KIND': 'terminus-2', 'TERMINUS': 'tb_agents.terminus_eval:Terminus2Eval',
                      'MODEL': 'hosted_vllm/' + c['served'], 'TEMPERATURE': str(p['temperature']), 'TOP_P': str(p['top_p']),
                      'HARBOR_EXTRA_ARGS': extra})
            return e
        e.update({'AGENT_KIND': 'pi-cont16k', 'MODEL': 'hosted_vllm/' + c['served'],
                  'PI_CONTEXT': ctx, 'PI_CONTEXT_WINDOW': ctx, 'PI_CONTEXT_PROXY': ctx,
                  'PI_MAX_OUTPUT': cap, 'PI_CONT16K_CAP': cap, 'PI_CONT16K_RESAMPLES': str(p['resamples']),
                  'PI_VERSION': p['pi_version'], 'PI_THINKING': p['pi_thinking'], 'PI_SAMPLING_SEED': str(s),
                  'PI_CONTEXT_POLICY': p['context_policy'], 'PI_RUNTIME_COMPAT_ARCHIVE': str(PI_RUNTIME_COMPAT),
                  'PI_CONT16K_CACHE': c.get('protocol', {}).get('pi_bundle_cache', str(PI_CACHE)), 'PI_KV_GATE': c.get('kv_gate', '0.80,0.72'), 'PI_GUARD_BOUND_SHORTCUT': '1',
                  'PI_GUARD_SUBMIT_AT_PI_CLAMP': '1', 'PI_HTTP_FORCE_CLOSE': '1', 'PI_KV_GATE_FILE': str(self.run / 'serve' / 'kv.json'),
                  'PI_TAIL_CUT_FILE': str(self.cut_file(k))})
        if self.INSTRUCTION_ARGV:   # tb_agents/harness_agents.py: "1" = the instruction as Pi's argument
            e['PI_INSTRUCTION_ARGV'] = '1'
        else:   # the worker's own environment never switches a stdin pipeline to the argument launch
            e.pop('PI_INSTRUCTION_ARGV', None)
        memcap_env(e, c, self.run / 'attempts' / k)
        diskcap_env(e, run_diskcap_config(self), self.run / 'attempts' / k, task)
        if c.get('reasoning_effort'):
            e['PI_REASONING_EFFORT'] = c['reasoning_effort']
        for x in ('PI_TEMPERATURE', 'PI_TOP_P'):   # only the run config sets them (never the worker's own environment)
            e.pop(x, None)
        e.update(temperature_env(c, p, attempt))
        # #375: the context guard's finish notice, frozen at init (missing = off; v3 runs without 'sticky' stay v3).
        # The worker's own environment never leaks one in: only the run config sets it.
        # PI_CONTEXT_RENDERER / PI_CONTEXT_TOKENIZER (aa-serving's registry hooks in the guard, renderer_hooks) are never
        # eval_q35's: the renderer they name (tb_agents/deepseek_v41_context.py) is on aa-evals only
        for x in ('PI_FINISH_WARN_AT', 'PI_FINISH_RESERVE', 'PI_FINISH_STICKY', 'PI_CONTEXT_RENDERER', 'PI_CONTEXT_TOKENIZER'):
            e.pop(x, None)
        e.update(finish_env(c))
        if c.get('ctx_gate'):   # context-aware turn gate (turn_gate.py): the proxy holds turns its key is not allowed
            e['PI_GATE_KEY'] = k
        if c.get('data_parallel_size', 1) > 1:   # keep each rollout on one vLLM DP rank (prefix cache reuse)
            e['PI_DP_RANK'] = str(dp_rank(k, c['data_parallel_size']))
        if c.get('envclock'):
            e['PI_CONT16K_CLASS'] = 'PiCont16kEnvClock'
            e['PI_ENVCLOCK_MULT'] = str(c.get('envclock_mult', 1))   # < 1 only for debugging the timeout path
            e['HARBOR_EXTRA_ARGS'] = f"--agent-timeout-multiplier {c['backstop_multiplier']:g}"
        return e

    async def run_item(self, k):
        task, attempt, retry = parse_key(k)
        root = self.run / 'attempts' / k
        raw = root / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        single = SCRATCH / 'single' / k
        single.mkdir(parents=True, exist_ok=True)
        if not (single / task).exists():
            (single / task).symlink_to(self.tasks_dir / task)
        endpoint = (self.run / 'serve' / 'endpoint.url').read_text().strip()
        env = self.env(k, task, attempt)
        (SCRATCH / 'tmp').mkdir(exist_ok=True)
        cmd = ['bash', str(REPO / 'gen/run_harbor.sh'), str(single), str(raw), k, endpoint, '1', '-k', '1',
               '--environment-build-timeout-multiplier', '4']
        info = {'key': k, 'task': task, 'attempt': attempt, 'retry': retry, 'seed': seed(task, attempt, self.config.get('seed_namespace', SEED_NAMESPACE)),
                'model': self.config['model'], 'served': self.config['served'], 'endpoint': endpoint,
                'worker': self.name, 'host': socket.gethostname(), 'started_at': now(), 'started_epoch': time.time(),
                'command': cmd, 'protocol': {**PROTOCOL, **self.config.get('protocol', {})}, 'envclock': bool(self.config.get('envclock')),
                'environment': {x: env[x] for x in env if x.startswith(('PI_', 'MAX_', 'HARBOR_EXTRA', 'AGENT_KIND', 'TERMINUS', 'TEMPERATURE', 'TOP_P'))},
                'diskcap': diskcap_config(self.diskcap_cfg, task),
                # the serving job behind this endpoint and its effective numerics hash (serve_numerics.py)
                'serving': serve_numerics.lookup(self.run, endpoint)}
        atomic(root / 'attempt.json', info)
        stopped = False
        with (root / 'harbor.log').open('ab', buffering=0) as output:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=output, stderr=asyncio.subprocess.STDOUT,
                                                        env=env, start_new_session=True)
            self.running[k]['proc'] = proc
            gone, cut_at = 0, None
            while proc.returncode is None:
                self.state.heartbeat(k)
                try:
                    await asyncio.wait_for(proc.wait(), 60)
                except asyncio.TimeoutError:
                    pass
                if STOP and proc.returncode is None:
                    stopped = True
                    await self.kill(proc)
                elif proc.returncode is None and self.cut_file(k).exists():
                    if cut_at is None:   # the agent stops itself (cut file); acknowledge so orchestrate waits
                        cut_at = time.time()
                        atomic(self.cut_file(k).with_suffix('.ack'), {'worker': self.name, 'at': now()})
                        log(f'{k}: tail cut; the agent stops and the verifier runs')
                    elif time.time() - cut_at > 2700:   # no clean stop within 45 min: end it (scored as a failure)
                        stopped = 'tail_cut'
                        await self.kill(proc)
                elif proc.returncode is None:
                    # The run's serving job moved (e.g. 4B H100 -> B200 takeover, or a restart elsewhere): the agent
                    # keeps calling the old endpoint (Pi's proxy waits out a dead backend for 30 min). Two checks a
                    # minute apart that it refuses connections end the attempt for a same-seed retry.
                    gone = gone + 1 if await asyncio.to_thread(self.endpoint_gone, endpoint) else 0
                    if gone >= 2:
                        log(f'{k}: endpoint {endpoint} gone (the run now serves at another endpoint); stopping')
                        stopped = 'endpoint_moved'
                        await self.kill(proc)
        outcome = classify(raw, proc.returncode, stopped if stopped != 'tail_cut' else True, self.config.get('harness', 'pi'),
                           agent_internet=self.config.get('agent_internet', True))
        cut = read(self.cut_file(k))
        if cut:   # tail rule: not an infrastructure failure, no retry; scored by the verifier when it ran
            scored = 'verifier' if stopped != 'tail_cut' and isinstance(outcome.get('reward'), (int, float)) else 'fail'
            outcome.update(infrastructure_failure=False, infrastructure_reason=None, tail_cut=True,
                           tail_cut_reason=cut.get('reason'), tail_cut_idle_h=cut.get('idle_h'), tail_cut_scored=scored,
                           score=float(outcome['reward']) if scored == 'verifier' else 0.0)
        outcome['wall_sec'] = round(time.time() - info['started_epoch'], 1)
        outcome.update(host=info.get('host'), worker=info.get('worker'))   # per-node failure rates (node_health.py)
        outcome.update(serving_numerics(self.run, info))
        box = read(root / 'memcap.json')
        if box:   # the main sandbox's peak (Pss anon + shmem) and the cap's kills (harbor_ns/memcap.py)
            outcome.update(sandbox_peak_mb=box.get('peak_mb'), memcap_limit_mb=box.get('limit_mb'),
                           memcap_kills=len(box.get('kills') or []))
        disk = read(root / 'diskcap.json')
        outcome['diskcap_limit_mb'] = None   # cap off, or the box never started
        if disk:   # the trial's overlay upper dirs: peak data, largest file, the cap's kills (harbor_ns/diskcap.py)
            outcome.update(sandbox_disk_peak_mb=disk.get('peak_mb'), sandbox_largest_file_mb=disk.get('largest_file_mb'),
                           diskcap_limit_mb=disk.get('limit_mb'), diskcap_kills=len(disk.get('kills') or []),
                           diskcap_at_fsize=len(disk.get('at_fsize') or []))
        if self.config.get('raw_layout', rawpack.LAYOUT_DEFAULT) == 'zip' and not STOP:
            # compact the finished attempt (shutdown skips it: the stop must stay fast; `rawpack.py pack-run` later)
            outcome = await asyncio.to_thread(rawpack.pack_attempt, root, outcome, SCRATCH / 'rawpack')
            if outcome['raw'].get('pack_error'):
                log(f"{k}: raw/ kept loose: {outcome['raw']['pack_error']}")
        atomic(root / 'outcome.json', outcome)
        if (self.state.q / 'done' / f'{k}.json').exists():
            log(f'{k}: already resolved by the stale-claim monitor; outcome kept in {root}/outcome.json only')
        else:
            self.state.finish(k, outcome)
        log(f"done {k}: score={outcome.get('score')} infra={outcome.get('infrastructure_reason')} "
            f"exc={outcome.get('exception_type')} {outcome['wall_sec']:.0f}s")

    def cut_file(self, k):
        """Tail rule (EVAL_REQUESTS.md 10-18 ~09:55 UTC): orchestrate.py writes this marker; the agent then stops as its
        time limit would (AgentTimeoutError) and Harbor's verifier scores the environment."""
        return self.run / 'queue' / 'cut' / f'{k}.json'

    def endpoint_gone(self, endpoint):
        try:
            current = (self.run / 'serve' / 'endpoint.url').read_text().strip()
        except OSError:
            return False   # no endpoint published right now (a restart in progress): keep waiting
        if current == endpoint:
            return False
        hostport = endpoint.split('://', 1)[-1].split('/', 1)[0]
        host, port = hostport.rsplit(':', 1)
        try:
            socket.create_connection((host, int(port)), timeout=10).close()
            return False
        except OSError:
            return True

    def claimable(self):
        claimed = self.state._names('claims')
        return any(f'{t}__a{k}__r0' not in claimed for t, k in self.state.items()) or bool(
            self.state._names('retry', '.json') - claimed)

    def stop_requested(self):
        """RUN/STOP_WORKERS stops claiming; 'drain <commit> ...' only for workers running those code snapshots."""
        try:
            words = (self.run / 'STOP_WORKERS').read_text().split()
        except FileNotFoundError:
            return False
        except OSError:
            return True
        if words[:1] == ['drain']:
            return any(REPO.name.startswith(c[:12]) for c in words[1:])
        return True

    @staticmethod
    async def kill(proc):
        for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(proc.wait(), wait)
                return
            except asyncio.TimeoutError:
                pass

    async def main(self):
        if self.RESOURCE_LOG:
            start_resource_log(self.run / 'workers' / f'{self.name}.resources.jsonl',
                               extra=lambda: {'running': len(self.running), 'slots': self.a.slots})
        available = self.stage()
        # by each staged task's effective cap too (an explicit --diskcap can make any task large)
        self.one_per_worker = diskcap_one_per_worker(self.diskcap_cfg, task_storage_mb(self.tasks_dir, available),
                                                     self.diskcap_worker_disk_mb)
        if self.one_per_worker:
            log(f'one attempt at a time per worker (disk cap): {sorted(self.one_per_worker)}')
            # for orchestrate.cut_tails (the tail rule's last-few cut waits for a held-back attempt of these)
            try:   # best effort: a transient /data error must not stop the worker (the config's list still applies)
                known = set((read(self.run / 'workers' / 'one_per_worker.json', {}) or {}).get('tasks') or ())
                if not self.one_per_worker <= known:
                    atomic(self.run / 'workers' / 'one_per_worker.json',
                           {'tasks': sorted(known | self.one_per_worker), 'updated_at': now()})
            except OSError as e:
                log(f'one_per_worker.json not written ({e!r})')
        log(f'{self.name}: staged {len(available)} tasks, {self.a.slots} slots')
        idle = 0
        while True:
            for k, item in list(self.running.items()):
                t = item['task']
                if t.done():
                    if t.exception():
                        log(f'{k} crashed: {t.exception()!r}')
                        self.state.finish(k, {'infrastructure_failure': True, 'infrastructure_reason': 'worker_exception',
                                              'score': None, 'exception_type': 'WorkerException',
                                              'exception_message': repr(t.exception())[:2000],
                                              'host': socket.gethostname(), 'worker': self.name})
                    del self.running[k]
            if STOP:
                if not self.running:
                    return
            elif (self.run / 'serve' / 'endpoint.url').exists() and not self.stop_requested():
                while len(self.running) < self.a.slots and await asyncio.to_thread(self.proxy_ok):
                    try:
                        k = await asyncio.to_thread(self.state.claim, self.name, self.claim_tasks(available))
                    except OSError as e:   # transient Lustre errors: try again next round
                        log(f'claim failed: {e!r}')
                        k = None
                    if k is None:
                        break
                    self.running[k] = {}
                    self.running[k]['task'] = asyncio.create_task(self.run_item(k))
                    log(f'claimed {k} ({len(self.running)} running)')
                    await asyncio.sleep(2)
            if not self.running and (self.state.complete() or self.stop_requested()):
                idle += 1
                if idle >= 3:
                    log('nothing left to run; exiting')
                    return
            elif not self.running and not self.claimable():
                idle += 1
                if idle * POLL_S >= self.IDLE_EXIT_S:   # nothing to claim: free the CPUs (orchestrate.py adds workers back)
                    log(f'no claimable attempts for {self.IDLE_EXIT_S // 60} min; exiting')
                    return
            else:
                idle = 0
            atomic(self.run / 'workers' / f'{self.name}.json', {'heartbeat': now(), 'running': sorted(self.running),
                                                                **self.heartbeat_extra()})
            await asyncio.sleep(POLL_S)

    def can_claim(self) -> bool:
        """Whether this worker would claim now if work were there (the main loop's conditions, proxy from its cache)."""
        return (not STOP and (self.run / 'serve' / 'endpoint.url').exists() and not self.stop_requested()
                and bool(self.proxy_state[1]))

    def heartbeat_extra(self) -> dict:
        """Heartbeat fields next to `running`: this worker's slots and whether it can claim now (aa_evals pi_eval
        claim packing, #531, counts only peers that can). Never raises: the heartbeat itself must be written."""
        try:
            return {'slots': int(self.a.slots), 'claiming': bool(self.can_claim())}
        except Exception:   # noqa: BLE001
            return {}


def serving_numerics(run, info):
    """Outcome fields naming the serving numerics of an attempt: the hash of the endpoint it started on and, when the
    run's endpoint changed while it ran (a move to another GPU type; the forwarder kept the old endpoint answering
    from the new server), the one it finished on, so a mid-run switch is visible per attempt and in score.py."""
    start = info.get('serving') or {}
    out = {'serving_numerics_sha256': start.get('numerics_sha256'), 'serving_job_id': start.get('job_id')}
    try:
        current = (Path(run) / 'serve' / 'endpoint.url').read_text().strip()
    except OSError:
        current = None
    if current and current != info.get('endpoint'):
        end = serve_numerics.lookup(run, current)
        out['serving_at_finish'] = end
        a, b = start.get('numerics_sha256'), end.get('numerics_sha256')
        out['serving_numerics_changed'] = None if a is None or b is None else a != b   # unrecorded: unknown, not changed
    return out


def run_diskcap_config(w):
    """A worker's run config as the disk-cap rules read it: without a "diskcap" key, the worker class's
    DISKCAP_WITHOUT_KEY decides between common.DISKCAP_DEFAULT (eval_q35) and off (pi_eval)."""
    if getattr(w, 'DISKCAP_WITHOUT_KEY', True) or 'diskcap' in w.config:
        return w.config
    return {**w.config, 'diskcap': None}


def memcap_env(e, config, attempt_dir):
    """harbor_ns per-sandbox memory cap from the run's config "memcap" (absent: off, as before #258)."""
    cap = config.get('memcap') or {}
    if cap.get('factor'):
        e['HARBOR_NS_MEMCAP'] = str(cap['factor'])
        e['HARBOR_NS_MEMCAP_MIN_MB'] = str(int(cap.get('min_mb') or 0))
        e['HARBOR_NS_MEMCAP_LOG'] = str(Path(attempt_dir) / 'memcap.json')
    return e


def task_storage_mb(tasks_dir, names):
    """{task: its task.toml [environment] storage_mb, or None} for staged tasks."""
    out = {}
    for t in names:
        try:
            out[t] = tomllib.loads((Path(tasks_dir) / t / 'task.toml').read_text()).get('environment', {}).get('storage_mb')
        except (OSError, ValueError):
            out[t] = None
    return out


def diskcap_env(e, config, attempt_dir, task=None):
    """harbor_ns per-sandbox disk cap env from the run's config (diskcap_config). Off is explicit (HARBOR_NS_DISKCAP=0), so
    a value in the worker's own environment cannot turn it on."""
    cap = diskcap_config(config, task)
    for k in ('HARBOR_NS_DISKCAP_MIN_MB', 'HARBOR_NS_DISKCAP_LOG'):
        e.pop(k, None)
    if not cap:
        e['HARBOR_NS_DISKCAP'] = '0'
        return e
    e['HARBOR_NS_DISKCAP'] = f"{cap['factor']:g}"
    e['HARBOR_NS_DISKCAP_MIN_MB'] = str(cap['min_mb'])
    e['HARBOR_NS_DISKCAP_LOG'] = str(Path(attempt_dir) / 'diskcap.json')
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--slots', type=int, default=32)
    a = ap.parse_args()

    def stop(sig, frame):
        global STOP
        STOP = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    asyncio.run(Worker(a).main())


if __name__ == '__main__':
    main()
