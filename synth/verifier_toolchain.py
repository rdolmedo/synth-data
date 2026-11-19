#!/usr/bin/env python3
"""Offline verifier toolchain for synth campaigns (RTS Harbor tasks). Port of eval_q35/verifier_toolchain.py.

Every RTS task's tests/test.sh fetches its pytest toolchain through the cluster proxy at verification time (apt curl,
the astral.sh uv installer, then uvx downloading a Python and wheels). The proxy has multi-hour outages: in the
2026-10-24 CalibForge hero campaign 42% of verifier runs scored 0 because these downloads failed. This module lets the
verifier run offline whenever that is equivalent:

  build:   (CPU HTCondor job with internet, synth/verifier_toolchain_job.sh) survey the tasks; per uv version the
           installer's own uv/uvx/env and the CPythons it would download; resolve once per unique (uv version, target
           Python, specs) group, download each unique pin once (parallel, proxy friendly) into one wheel pool; offline
           checks per group (offline resolution == online resolution, offline installs); manifest.json. On /data the
           toolchain is a few thousand files: Pythons are one tarball each, one wheel pool.
  worker:  prepare_verifier(task_dir, toolchain_root, local_cache) right before an attempt: the wrapper becomes
           tests/test.sh (the task's script is tests/test_task.sh) and, for an offline task, tests/.toolchain (copied
           lazily from /data into a node-local cache once, then hardlinked) plus tests/test_offline.sh. Harbor uploads
           tests/ only at verification, so agents never see either. release_verifier() after the attempt.
  runtime: VERIFIER_WRAPPER runs test_offline.sh first (uv from the toolchain, UV_OFFLINE, wheels from find-links,
           Pythons from the toolchain only where the image has none, like an online run); if pytest did not start it
           runs the task's own script online, retrying its network bootstrap within half the verifier timeout.
  validate/compare/finalize: Harbor's oracle agent; offline then online verifier on the same final state.

Integrity rules (kept from the eval module): apt/yum/dnf/apk lines are dropped only when they install nothing but curl
(+ ca-certificates) for the uv installer and no test touches curl, openssl, certificates, https, apt or the installer's
shell-profile edits; other package lines stay, guarded (a failure defers to the online script). Tasks whose tests run
uv/pip at test time, whose test.sh runs other network installs, uses uv beyond uvx, non-index or torch-class specs,
or changed since the build stay online.

  python -m synth.verifier_toolchain survey  --tasks-tar T.tar --order order.txt --limit 16000
  python -m synth.verifier_toolchain build   --tasks-tar T.tar --order order.txt --limit 16000 --out ROOT
  python -m synth.verifier_toolchain validate ...   (see synth/verifier_toolchain_job.sh)
"""
import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

TOOLCHAIN = Path('/data/aai/tbench/verifier_toolchain/rts-v1')
BUNDLE = 'rts'
LABEL = '[synth verifier]'

ASTRAL = re.compile(r'^[ \t]*curl[^\n|]*https://astral\.sh/uv/([0-9][0-9.]*)/install\.sh[ \t]*\|[ \t]*(?:sh|bash)\b[^\n]*$', re.M)
CURL_WORD = re.compile(r'(?<![\w.-])curl(?![\w.-])')
UV_PROJECT = re.compile(r'\buv (?:add|lock|sync)\b')
# a test that itself runs uv / pip would inherit the offline settings (UV_OFFLINE, find-links): such tasks stay online
TEST_TIME_INSTALL = re.compile(r'(?:^|[^A-Za-z_.-])(?:uvx|uv (?:pip|run|add|venv|tool|sync)|pip3? install|python3? -m pip)\b'
                               # subprocess list forms: ["pip", "install", ...], [sys.executable, "-m", "pip", ...]
                               r'|[\'"]pip3?[\'"]\s*,\s*[\'"](?:install|uninstall|download|wheel)[\'"]'
                               r'|[\'"]-m[\'"]\s*,\s*[\'"]pip[\'"]'
                               r'|[\'"]uvx?[\'"]\s*,\s*[\'"](?:pip|run|add|sync|lock|tool|venv)[\'"]')
# apt-get install -y curl also pulls ca-certificates, openssl, libcurl: tests that may touch those (or apt state) keep
# their package lines in the offline script
APT_SENSITIVE = re.compile(r'openssl|ca-certificates|update-ca|/etc/ssl|certifi|\bssl\b|https://(?!(?:localhost|127\.0\.0\.1))|'
                           r'libcurl|pycurl|\bwget\b|apt-get|\bapt\b|dpkg|apt-cache|\byum\b|\bdnf\b|\brpm\b|\bapk\b', re.I)
# the uv installer appends a PATH line to the shell profiles and writes ~/.local/bin and ~/.config/uv: tests reading
# those keep the online path (the offline script only copies uv/uvx/env into ~/.local/bin)
INSTALLER_SENSITIVE = re.compile(r'bashrc|bash_profile|bash_login|(?<![\w-])\.profile\b|zshrc|zshenv|\.local/bin|'
                                 r'uv-receipt|config/fish|\.config/uv', re.I)
# commands in test.sh (outside the bootstrap) that need the network on their own: such tasks stay online
NETWORK_CMD = re.compile(r'\bpip3? install\b|-m pip\b|\bnpm (?:install|ci|i)\b|\byarn (?:add|install)\b|\bgit clone\b|'
                         r'\bwget\b|\bgo (?:get|install|mod download)\b|\bcargo (?:install|fetch)\b|\bgem install\b|'
                         r'\bbundle install\b|\bcomposer (?:install|require)\b|\bconda install\b|\bnpx\b')
HEAVY = re.compile(r'^(torch|torchvision|torchaudio|tensorflow|jax|jaxlib|nvidia-|triton|vllm|xformers)\b', re.I)
# Pythons a bare `uvx` may run on: the image's (3.8 .. 3.13 on the RTS bases) or the uv version's default download
TARGETS_IMPLICIT = ['3.8', '3.9', '3.10', '3.11', '3.12', '3.13']
PER_TASK_CAP = 500 * 2 ** 20
# wheels for the oldest glibc first: the same pinned version installs on centos:7 / ubuntu:18.04 bases too
PLATFORMS = ['manylinux1_x86_64', 'manylinux2010_x86_64', 'manylinux2014_x86_64'] + \
            [f'manylinux_2_{x}_x86_64' for x in range(17, 32)]
TEST_FILE_SUFFIXES = ('', '.py', '.sh', '.js', '.ts', '.rb', '.pl', '.txt', '.json', '.yaml', '.yml', '.cfg', '.toml',
                      '.ini', '.conf')
OWN_SCRIPTS = ('test.sh', 'test_task.sh', 'test_offline.sh', 'test_offline_strict.sh')


def say(tag, msg):
    print(f'[{tag}] {time.strftime("%H:%M:%SZ", time.gmtime())} {msg}', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---- parse (what a task's verifier downloads) --------------------------------------------------------------------

def test_files(task_dir):
    """The task's test files (not its test scripts, not a staged .toolchain)."""
    out = []
    for root, dirs, files in os.walk(Path(task_dir) / 'tests'):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.toolchain'))
        for name in sorted(files):
            f = Path(root) / name
            if name not in OWN_SCRIPTS and f.suffix in TEST_FILE_SUFFIXES and f.is_file():
                out.append(f)
    return out


def scan_tests(task_dir, pattern):
    hits = []
    for f in test_files(task_dir):
        try:
            if pattern.search(f.read_text(errors='replace')):
                hits.append(f.name)
        except OSError:
            pass
    return hits


def test_time_install(task_dir):
    """Test files (other than the test scripts) that call uv / uvx / pip."""
    return scan_tests(task_dir, TEST_TIME_INSTALL)


HEREDOC = re.compile(r'<<-?\s*([\'"]?)([A-Za-z_][A-Za-z0-9_]*)\1')


def written_curl(text):
    """curl inside what a shell script writes to files (heredoc bodies, echo/printf redirected to a file): a script
    the tests may run later. curl run by the script itself (solve time, agent phase) does not count."""
    lines, i = text.split('\n'), 0
    while i < len(lines):
        line = lines[i]
        m = HEREDOC.search(line)
        if m and not is_comment(line):
            j, body = i + 1, []
            while j < len(lines) and lines[j].strip() != m.group(2):
                body.append(lines[j])
                j += 1
            if CURL_WORD.search('\n'.join(body)):
                return True
            i = j + 1
            continue
        if re.search(r'\b(?:echo|printf)\b.*(?<![\w.-])curl(?![\w.-]).*>', line):
            return True
        i += 1
    return False


def apt_sensitive(task_dir):
    """What may observe the dropped curl install (curl, libcurl, TLS certificates, openssl, apt/dpkg state, the
    installer's profile edits): test files matching APT_SENSITIVE / curl / INSTALLER_SENSITIVE, an instruction that
    mentions curl, and reference-solution files that write curl into scripts (the tests may run them)."""
    task_dir = Path(task_dir)
    hits = scan_tests(task_dir, APT_SENSITIVE) + [f'{h} (curl)' for h in scan_tests(task_dir, CURL_WORD)] + \
        [f'{h} (installer)' for h in scan_tests(task_dir, INSTALLER_SENSITIVE)]
    try:
        if re.search(r'\bcurl\b', (task_dir / 'instruction.md').read_text(errors='replace'), re.I):
            hits.append('instruction.md (curl)')
    except OSError:
        pass
    sol = task_dir / 'solution'
    if sol.is_dir():
        for f in sorted(sol.rglob('*')):
            try:
                if not f.is_file() or f.stat().st_size > 2 ** 20:
                    continue
                text = f.read_text(errors='replace')
                if (written_curl(text) if f.suffix in ('.sh', '') else CURL_WORD.search(text)):
                    hits.append(f'solution/{f.name} (curl)')
            except OSError:
                pass
    return sorted(set(hits))


def logical_lines(text):
    """[(first, last, joined)] physical line ranges of each logical line (backslash continuations joined). Heredoc
    bodies are returned line by line with joined = '' (never parsed or rewritten)."""
    lines = text.split('\n')
    out, i, heredoc = [], 0, None
    while i < len(lines):
        if heredoc is not None:
            out.append((i, i, ''))
            if (lines[i].lstrip('\t') if heredoc[1] else lines[i]) == heredoc[0]:
                heredoc = None
            i += 1
            continue
        j, parts = i, [lines[i]]
        while parts[-1].endswith('\\') and j + 1 < len(lines):
            parts[-1] = parts[-1][:-1]
            j += 1
            parts.append(lines[j].lstrip())
        joined = ' '.join(p.rstrip() if k < len(parts) - 1 else p for k, p in enumerate(parts))
        out.append((i, j, joined))
        m = HEREDOC_START.search(joined)
        if m and not is_comment(joined):
            heredoc = (m.group(3), m.group(1) == '-')
        i = j + 1
    return out


def is_comment(line):
    return line.lstrip().startswith('#')


HEREDOC_START = re.compile(r'<<(-?)\s*([\'"]?)([A-Za-z_][A-Za-z0-9_]*)\2')
REDIRECT = re.compile(r'\s*(?:&>>?|[12]?>>?&?)\s*(?:/dev/null|&?[12])')
PM_WORD = re.compile(r'(?:^|[\s;&|(`])(?:sudo\s+)?(?:apt-get|apt|yum|dnf|microdnf|apk|zypper|pacman)\s')
CHECK_ONLY = re.compile(r'\b(?:command\s+-v|which|type|hash)\s+\S+')
CURL_PKGS = {'curl', 'ca-certificates'}


def classify_segment(seg):
    """One simple command of an and-or list: ('update'|'install'|'clean'|'check'|'true'|'other', packages)."""
    s = REDIRECT.sub('', seg).strip()
    s = re.sub(r'^(?:sudo\s+)?(?:DEBIAN_FRONTEND=\S+\s+)+', '', s)
    s = re.sub(r'^sudo\s+', '', s)
    toks = s.split()
    if not toks:
        return 'true', ()
    if toks in (['true'], [':']):
        return 'true', ()
    if s in ('rm -rf /var/lib/apt/lists/*', 'rm -rf /var/cache/yum', 'rm -rf /var/cache/dnf', 'rm -rf /var/cache/apk/*'):
        return 'clean', ()
    m = re.fullmatch(r'(?:!\s*)?(?:command\s+-v|which|type|hash)\s+(\S+)', s)
    if m:
        return 'check', (m.group(1),)
    pm, rest = toks[0], toks[1:]
    opts = [t for t in rest if t.startswith('-')]
    words = [t for t in rest if not t.startswith('-')]
    # option values (apt-get -o X=Y) are words too: only the known flag-only options keep a line classifiable
    flag_ok = {'-y', '-yq', '-qy', '-q', '-qq', '-yqq', '--yes', '--quiet', '--no-install-recommends', '--no-install-suggests',
               '--allow-unauthenticated', '--no-cache', '--nogpgcheck', '--setopt=install_weak_deps=False', '--allowerasing',
               '--assumeyes'}
    if pm in ('apt-get', 'apt') and words[:1] == ['update'] and len(words) == 1 and all(o in flag_ok for o in opts):
        return 'update', ()
    if pm in ('apt-get', 'apt') and words[:1] == ['update'] and opts and words[1:] and all(
            o in flag_ok or o == '-o' for o in opts) and all('=' in w for w in words[1:]):
        return 'update', ()   # apt-get update -o Acquire::...=true
    if pm in ('apt-get', 'apt', 'yum', 'dnf', 'microdnf', 'zypper') and words[:1] == ['install'] and words[1:] \
            and all(o in flag_ok for o in opts):
        return 'install', tuple(words[1:])
    if pm == 'apk' and words[:1] == ['add'] and words[1:] and all(o in flag_ok for o in opts):
        return 'install', tuple(words[1:])
    if (pm in ('yum', 'dnf', 'microdnf') and words == ['clean', 'all']) or (pm in ('apt-get', 'apt') and words == ['clean']):
        return 'clean', ()
    return 'other', ()


def split_and_or(line):
    """'a && b || c' -> (['a', 'b', 'c'], ['&&', '||']), or None when the line is not a plain and-or list (pipes,
    ';', background, compound commands, quotes, substitutions)."""
    if re.search(r'[;`"\'(){}]|\$\(|(?<![&|>])\|(?!\|)|(?<![&>])&(?![&>])|^\s*(?:if|then|else|elif|fi|for|while|do|done|case)\b',
                 line):
        return None
    parts = re.split(r'(&&|\|\|)', line)
    segs, ops = parts[0::2], parts[1::2]
    if any(not s.strip() for s in segs):
        return None
    return [s.strip() for s in segs], ops


def package_line(line):
    """For a logical line that runs a package manager: {'strippable': it only installs curl (+ ca-certificates) for the
    uv installer, 'guardable': a plain and-or list a guard can be appended to, 'packages': [...]}; None otherwise."""
    if is_comment(line) or not PM_WORD.search(' ' + CHECK_ONLY.sub('', line)):
        return None
    sp = split_and_or(line.strip())
    if sp is None:
        return {'strippable': False, 'guardable': False, 'packages': []}
    segs, _ = sp
    kinds = [classify_segment(s) for s in segs]
    pkgs = sorted({p for k, ps in kinds if k == 'install' for p in ps})
    curl_only = all(k in ('update', 'clean', 'true') or (k == 'install' and set(ps) <= CURL_PKGS and 'curl' in ps)
                    or (k == 'check' and ps == ('curl',)) for k, ps in kinds)
    return {'strippable': curl_only, 'guardable': True, 'packages': pkgs}


UV_INVOKE = re.compile(r'(?:^|(?<=[\s;&|(`]))(?:"?\$\{?HOME\}?/\.local/bin/|~/\.local/bin/|/root/\.local/bin/)?(uvx|uv)"?(?=\s|$)')
UVX_VALUE = {'-p', '--python', '-w', '--with', '--from', '--with-requirements'}
UVX_FLAGS = {'-q', '--quiet', '-v', '--verbose', '--isolated', '--no-cache', '-n', '--no-progress', '--offline'}
UVX_ONLINE = {'-i', '--index-url', '--extra-index-url', '--index', '--default-index', '-f', '--find-links',
              '--with-editable', '--python-preference', '--no-index', '--index-strategy', '--prerelease',
              '--exclude-newer', '--resolution', '-c', '--constraint', '--overrides'}


def shell_words(rest):
    """Words of a command line up to the first control operator / redirection, or None when unparseable."""
    try:
        lx = shlex.shlex(rest, posix=True, punctuation_chars=';&|()<>')
        lx.whitespace_split = True
        out = []
        for tok in lx:
            if tok and set(tok) <= set(';&|()<>'):
                break
            out.append(tok)
        return out
    except ValueError:
        return None


def python_minor(v):
    m = re.fullmatch(r'(?:python|cpython)?([0-9]+\.[0-9]+)(?:\.[0-9]+)?', v)
    return m.group(1) if m else None


def parse_spec(s):
    """A PEP 508 requirement without URL or marker (uv tool syntax name@version -> name==version), or None."""
    from packaging.requirements import InvalidRequirement, Requirement
    if '$' in s or '://' in s or s.startswith(('git+', '.', '/')) or ',' in s and not re.search(r'[<>=!~]', s):
        return None
    m = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?)@([0-9][A-Za-z0-9.+!-]*)', s)
    if m:
        s = f'{m.group(1)}=={m.group(2)}'
    try:
        r = Requirement(s)
    except InvalidRequirement:
        return None
    if r.url or r.marker:
        return None
    return s


def uv_commands(text):
    """[(verb, words)] for every uv / uvx invocation in a script (logical lines, comments and echo lines skipped)."""
    out = []
    for _, _, line in logical_lines(text):
        if is_comment(line) or re.match(r'\s*(?:echo|printf)\b', line):
            continue
        for m in UV_INVOKE.finditer(line):
            before = line[:m.start()]
            if re.search(r'(?:command\s+-v|which|type|hash|-[fexs]|\[)\s+"?$', before):
                continue
            words = shell_words(line[m.end():])
            out.append((m.group(1), words))
    return out


def parse_uvx(words, task_dir):
    """uvx [options] TOOL args: (specs, pythons, implicit, tool, reason)."""
    specs, pythons, tool, k = [], [], None, 0
    while k < len(words):
        w = words[k]
        name, eq, val = w.partition('=')
        if w == '--':
            k += 1
            tool = words[k] if k < len(words) else None
            break
        if name in UVX_ONLINE:
            return None, None, None, None, f'uvx option {name}'
        if name in UVX_VALUE:
            if not eq:
                if k + 1 >= len(words):
                    return None, None, None, None, f'uvx option {name} without value'
                val = words[k + 1]
                k += 1
            if name in ('-p', '--python'):
                py = python_minor(val)
                if not py:
                    return None, None, None, None, f'uvx python request {val!r}'
                pythons.append(py)
            elif name == '--with-requirements':
                p = Path(task_dir) / 'tests' / Path(val).name if val.startswith('/tests/') or '/' not in val else None
                if p is None or not p.exists():
                    return None, None, None, None, f'requirements file {val} not in tests/'
                specs += [l.split('#')[0].strip() for l in p.read_text().splitlines() if l.split('#')[0].strip()]
            else:
                specs.append(val)
            k += 1
            continue
        if w in UVX_FLAGS:
            k += 1
            continue
        if w.startswith('-'):
            return None, None, None, None, f'uvx option {w}'
        tool = w
        break
    if not tool:
        return None, None, None, None, 'uvx without a tool'
    base = re.split(r'[@=<>!~\[]', tool, maxsplit=1)[0].lower()
    if base == 'python':
        return None, None, None, None, f'uvx {tool} (interpreter request)'
    specs.append(tool)
    return specs, pythons, not pythons, base, None


def parse(task_dir):
    """What a task's verifier downloads and whether an offline run is equivalent. kind: 'uv' (pinned astral uv
    installer: offline candidate unless reason is set) or 'none' (no pinned installer: the task is left unchanged)."""
    task_dir = Path(task_dir)
    tests = task_dir / 'tests'
    p = tests / ('test_task.sh' if (tests / 'test_task.sh').exists() else 'test.sh')
    info = {'task': task_dir.name, 'kind': 'none', 'reason': None}
    if not p.exists():
        info['reason'] = 'no tests/test.sh'
        return info
    text = p.read_text(errors='replace')
    info['test_sh_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    uv_versions = sorted(set(ASTRAL.findall(text)))
    if not uv_versions:
        info['reason'] = 'unpinned uv installer (latest at run time)' if 'astral.sh/uv/install.sh' in text \
            else 'no pinned astral uv installer line'
        return info
    info['kind'] = 'uv'
    if len(uv_versions) > 1:
        info['reason'] = f'several uv versions {uv_versions}'
        return info
    info['uv_version'] = uv_versions[0]
    reasons = []
    lls = logical_lines(text)
    installer_at = next((i for i, (_, _, l) in enumerate(lls) if ASTRAL.search(l)), None)
    if installer_at is None:
        info['reason'] = 'uv installer inside a continued line'
        return info
    # package-manager lines: curl-only bootstrap lines before the installer may be dropped, others stay guarded
    pkg_lines = [(i, package_line(l)) for i, (_, _, l) in enumerate(lls)]
    pkg_lines = [(i, pl) for i, pl in pkg_lines if pl]
    if any(not pl['guardable'] for _, pl in pkg_lines):
        reasons.append('package-manager line in a compound command')
    first_uv = next((i for i, (_, _, l) in enumerate(lls) if not is_comment(l) and UV_INVOKE.search(l)
                     and i > installer_at), None)
    if first_uv is not None and any(i > first_uv for i, _ in pkg_lines):
        reasons.append('package-manager line after the tests start')
    info['packages'] = sorted({p for _, pl in pkg_lines for p in pl['packages']})
    # curl / network commands after the installer line, in test.sh itself
    # (raw lines, heredoc bodies included: a script the test writes and runs counts)
    after = '\n'.join(l for l in text.split('\n')[lls[installer_at][1] + 1:] if not is_comment(l))
    info['curl_after_bootstrap'] = bool(CURL_WORD.search(CHECK_ONLY.sub('', after)))
    net = sorted({m.group(0) for m in NETWORK_CMD.finditer('\n'.join(l for l in text.split('\n') if not is_comment(l)))})
    if net:
        reasons.append(f'test.sh runs other network commands {net[:3]}')
    if re.search(r'(?m)^(?!\s*#).*\bUV_[A-Z_]+=', text):
        reasons.append('test.sh sets UV_* variables')
    if UV_PROJECT.search(text):
        reasons.append('uv add/lock/sync (universal resolution needs the index)')
    tti = test_time_install(task_dir)
    if tti:
        reasons.append(f'tests call uv/pip at test time ({tti[:3]})')
    # uv commands: only uvx / uv tool run with index packages
    specs, pythons, implicit, tools, commands = [], set(), False, set(), set()
    cmds = uv_commands(text)
    for prog, words in cmds:
        if words is None:
            reasons.append('unparseable uv command line')
            continue
        if prog == 'uv':
            if words[:2] == ['tool', 'run']:
                words = words[2:]
            elif words[:1] == ['--version'] or words[:2] == ['--version'] or not words:
                continue
            else:
                reasons.append(f'uv {" ".join(words[:2])} (stays online)')
                continue
        sp, py, imp, tool, why = parse_uvx(words, task_dir)
        if why:
            reasons.append(why)
            continue
        bad = [s for s in sp if parse_spec(s) is None]
        if bad:
            reasons.append(f'non-index specs {bad[:3]}')
            continue
        specs += [parse_spec(s) for s in sp]
        commands.add((tuple(sorted({parse_spec(s) for s in sp})), tuple(sorted(py))))
        pythons.update(py)
        implicit |= imp
        tools.add(tool)
    if not cmds:
        reasons.append('no uvx command')
    if len(commands) > 1:
        # each uvx command resolves on its own: the group's union resolution could pin other versions
        reasons.append(f'{len(commands)} uvx commands with different packages or Pythons')
    if tools and tools != {'pytest'}:
        reasons.append(f'uvx tool {sorted(tools)} (the wrapper detects pytest sessions only)')
    info['specs'] = sorted(set(specs))
    info['pythons'] = sorted(pythons)
    info['implicit_python'] = implicit
    heavy = [s for s in info['specs'] if HEAVY.match(s)]
    if heavy:
        reasons.append(f'heavy dependency {heavy[:3]} (stays online)')
    sens = apt_sensitive(task_dir)
    info['apt_sensitive'] = sens
    # strip_apt: the offline script needs no package manager (its package lines, if any, only fetched curl)
    info['strip_apt'] = all(pl['strippable'] and i < installer_at for i, pl in pkg_lines) \
        and not info['curl_after_bootstrap'] and not sens
    if reasons:
        info['reason'] = reasons[0]
        info['reasons'] = reasons
    return info


def group_key(rec):
    """Tasks sharing a resolution: same uv version, Python requests and specs."""
    return json.dumps([rec['uv_version'], rec['pythons'], rec['implicit_python'], rec['specs']])


def group_id(key):
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---- task sources --------------------------------------------------------------------------------------------------

def extract_tasks(tars, names, dst):
    """Extract tests/, solution/, task.toml, instruction.md of the named tasks from task tars (members
    tasks/<id>/...), skipping environment/ (the build never builds images)."""
    names, dst = set(names), Path(dst)
    found = set()
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf:
                parts = Path(m.name).parts
                if len(parts) < 3 or parts[0] != 'tasks' or parts[1] not in names or parts[2] == 'environment':
                    continue
                if not (m.isfile() or m.isdir()):
                    continue
                tf.extract(m, dst, filter='data')
                found.add(parts[1])
    return found


def _parse_one(path):
    try:
        return Path(path).name, parse(path)
    except Exception as e:   # noqa: BLE001  (a parser bug must not stop a 16k-task build; the task stays online)
        return Path(path).name, {'task': Path(path).name, 'kind': 'none', 'reason': f'parse error {e!r}'[:300]}


def survey(tasks_dir, names, bundle=BUNDLE, jobs=8):
    paths = [str(Path(tasks_dir) / n) for n in names]
    with ProcessPoolExecutor(jobs) as ex:
        return {f'{bundle}/{name}': rec for name, rec in ex.map(_parse_one, paths, chunksize=64)}


def task_names(order, limit):
    names = [l.strip() for l in Path(order).read_text().splitlines() if l.strip()]
    return names[:limit] if limit else names


# ---- build (CPU job with internet) ---------------------------------------------------------------------------------

BUILD_NET_FAIL = re.compile(r'network|proxy|connect|resolve|timed out|timeout|reset|curl: \(|error sending request|'
                            r'failed to (fetch|download)|temporarily unavailable|503|502|504', re.I)


def run(cmd, env=None, cwd=None, check=True, timeout=3600, retries=0, backoff=30):
    """Run a command; `retries` more tries (backoff x try, seconds) while it fails with a network-looking error (the
    cluster proxy resets connections for minutes at a time: the first 30k-task build died on the uv installer)."""
    for attempt in range(retries + 1):
        try:
            r = subprocess.run([str(c) for c in cmd], env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            r = subprocess.CompletedProcess(e.cmd, 124, str(e.stdout or ''), f'timeout after {timeout}s')
        if r.returncode == 0 or attempt == retries or not BUILD_NET_FAIL.search(r.stderr + r.stdout):
            break
        say('build', f'retry {attempt + 1}/{retries} in {backoff * (attempt + 1)}s: {" ".join(map(str, cmd))[:120]}')
        time.sleep(backoff * (attempt + 1))
    if check and r.returncode != 0:
        raise RuntimeError(f'{" ".join(map(str, cmd))} -> {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    return r


def base_env(**extra):
    return {**{k: v for k, v in os.environ.items() if not k.startswith('UV_')}, **{k: str(v) for k, v in extra.items()}}


def relativize_links(root):
    """uv's minor-version links are absolute: make every absolute symlink under root relative."""
    for p in list(Path(root).rglob('*')):
        if p.is_symlink():
            t = os.readlink(p)
            if os.path.isabs(t):
                os.remove(p)
                os.symlink(os.path.relpath(t, p.parent), p)


def installed_pythons(pyd):
    """{minor: install dir name} of the uv-managed CPythons in pyd (minor-version links skipped)."""
    out = {}
    for d in sorted(Path(pyd).iterdir()) if Path(pyd).exists() else []:
        m = re.match(r'cpython-(\d+\.\d+)\.\d+-linux-x86_64-gnu$', d.name)
        if m and d.is_dir() and not d.is_symlink():
            out[m.group(1)] = d.name
    return out


def norm_pin(pin):
    name, _, ver = pin.partition('==')
    return f'{re.sub(r"[-_.]+", "-", name).lower()}=={ver.strip()}'


def wheel_meta(path):
    """{bytes, sha256, project, requires_python} of a wheel (for the local simple index)."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        meta = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA') and n.count('/') == 1)
        text = z.read(meta).decode('utf-8', 'replace').split('\n\n', 1)[0]
    name = re.search(r'^Name:\s*(.+)$', text, re.M).group(1).strip()
    rp = re.search(r'^Requires-Python:\s*(.+)$', text, re.M)
    return {'bytes': Path(path).stat().st_size, 'sha256': sha256(path), 'project': re.sub(r'[-_.]+', '-', name).lower(),
            'requires_python': rp.group(1).strip() if rp else None}


def write_simple_index(dst, wheels, meta, href='../../wheels/'):
    """A PEP 503 index of the given wheels (dst/<project>/index.html, links to href<wheel>). Unlike flat find-links,
    an index tells uv each file's requires-python up front, so an image Python too old for the pins makes uv refine
    the interpreter request exactly as it does online (and pick the toolchain's Python)."""
    import html
    projects = collections.defaultdict(list)
    for w in sorted(wheels):
        projects[meta[w]['project']].append(w)
    dst = Path(dst)
    for proj, files in sorted(projects.items()):
        (dst / proj).mkdir(parents=True, exist_ok=True)
        links = ''.join(f'<a href="{href}{w}#sha256={meta[w]["sha256"]}"'
                        + (f' data-requires-python="{html.escape(meta[w]["requires_python"])}"' if meta[w]['requires_python'] else '')
                        + f'>{w}</a>\n' for w in files)
        (dst / proj / 'index.html').write_text(f'<!DOCTYPE html>\n<html><body>\n{links}</body></html>\n')
    (dst / 'index.html').write_text('<!DOCTYPE html>\n<html><body>\n' + ''.join(
        f'<a href="{p}/">{p}</a>\n' for p in sorted(projects)) + '</body></html>\n')


def tool_dir(root):
    """A PATH directory with the coreutils console scripts need (realpath, dirname: long-shebang trampolines) and no
    Python, so the image-Python scenarios control which interpreters uv can discover."""
    d = Path(root) / 'tools'
    d.mkdir(parents=True, exist_ok=True)
    for n in ('sh', 'bash', 'env', 'realpath', 'dirname', 'readlink', 'uname', 'cat', 'ls', 'mkdir', 'rm', 'tr', 'sed'):
        src = shutil.which(n, path='/usr/bin:/bin')
        if src and not (d / n).exists():
            os.symlink(src, d / n)
    return d


PROBE = '''import importlib.metadata as m, json, sys
def test_probe():
    print("PROBE " + json.dumps({"python": "%d.%d.%d" % sys.version_info[:3], "dists": sorted(
        d.metadata["Name"] + "==" + d.version for d in m.distributions())}))
'''


def build(out, tars, order, limit, work, bundle=BUNDLE, jobs=8, threads=16, reuse=None):
    """Build the toolchain for the first `limit` tasks of `order` (task tars: members tasks/<id>/...) in work (local
    disk), then copy it to out. reuse: an earlier toolchain root; its offline groups (same uv version and default
    Python) keep their resolution, wheels and checks, and its downloads are not repeated, so extending a build to more
    tasks only resolves, downloads and checks the new groups. Returns the manifest."""
    work = Path(work)
    old = json.loads((Path(reuse) / 'manifest.json').read_text()) if reuse else None
    work.mkdir(parents=True, exist_ok=True)
    names = task_names(order, limit)
    tasks_dir = work / 'tasks'
    if not (work / 'tasks.READY').exists():
        found = extract_tasks(tars, names, work)
        (work / 'tasks.READY').write_text(json.dumps(sorted(found)))
    found = set(json.loads((work / 'tasks.READY').read_text()))
    say('build', f'{len(found)}/{len(names)} tasks extracted from {len(tars)} tars')
    tasks = survey(tasks_dir, [n for n in names if n in found], bundle, jobs)
    for n in names:
        if n not in found:
            tasks[f'{bundle}/{n}'] = {'task': n, 'kind': 'none', 'reason': 'not in the task tars'}
    cands = {k: r for k, r in tasks.items() if r['kind'] == 'uv' and not r.get('reason')}
    say('build', f'{len(tasks)} tasks surveyed, {len(cands)} offline candidates')
    tc = work / 'tc'
    (tc / 'pool').mkdir(parents=True, exist_ok=True)
    uv_versions = sorted({r['uv_version'] for r in cands.values()}, key=lambda v: [int(x) for x in v.split('.')])
    cache = work / 'uv-cache'
    env = base_env(HOME=work / 'home', UV_CACHE_DIR=cache)
    (work / 'home').mkdir(exist_ok=True)
    uvbin, defaults = {}, {}
    for v in uv_versions:
        home = work / f'home-{v}'
        home.mkdir(exist_ok=True)
        # The eval toolchain holds the same installer outputs for some versions: no proxy round trip for those.
        seed = Path('/data/aai/tbench/verifier_toolchain/v1/uv') / v
        if all((seed / f).is_file() for f in ('uv', 'uvx', 'env')):
            (home / '.local' / 'bin').mkdir(parents=True, exist_ok=True)
            for f in ('uv', 'uvx', 'env'):
                shutil.copy2(seed / f, home / '.local' / 'bin' / f)
        # `curl | sh` exits 0 when the download breaks (sh runs a truncated script): pipefail, then check the binary
        for attempt in range(9):
            if all((home / '.local' / 'bin' / f).exists() for f in ('uv', 'uvx', 'env')) and run(
                    [home / '.local' / 'bin' / 'uv', '--version'], check=False).stdout.split()[1:2] == [v]:
                break
            r = run(['bash', '-c', f'set -o pipefail; curl -LsSf https://astral.sh/uv/{v}/install.sh | sh'],
                    env=base_env(HOME=home), check=False, retries=0)
            ok = all((home / '.local' / 'bin' / f).exists() for f in ('uv', 'uvx', 'env')) and run(
                [home / '.local' / 'bin' / 'uv', '--version'], check=False).stdout.split()[1:2] == [v]
            if ok:
                break
            if attempt == 8:
                raise RuntimeError(f'uv {v} installer failed 9 times: {r.stderr[-1000:]}')
            say('build', f'uv {v} installer try {attempt + 1} failed (rc {r.returncode}); retrying in {30 * (attempt + 1)}s')
            time.sleep(30 * (attempt + 1))
        d = tc / 'uv' / v
        d.mkdir(parents=True, exist_ok=True)
        for f in ('uv', 'uvx', 'env'):
            shutil.copy2(home / '.local' / 'bin' / f, d / f)
        uvbin[v] = d / 'uv'
        # the CPython a bare request downloads with this uv version (when the image has none)
        dd = work / 'python-default' / v
        if len(installed_pythons(dd)) != 1:
            shutil.rmtree(dd, ignore_errors=True)
            run([uvbin[v], 'python', 'install'], retries=8, env={**env, 'UV_PYTHON_INSTALL_DIR': str(dd),
                                                       'UV_PYTHON_BIN_DIR': str(work / f'pybin-{v}')})
        minors = sorted(installed_pythons(dd))
        if len(minors) != 1:
            raise RuntimeError(f'uv {v}: default Python install left {minors}')
        defaults[v] = minors[0]
        name = installed_pythons(dd)[minors[0]]
        if not (work / 'python' / v / name).exists():
            (work / 'python' / v).mkdir(parents=True, exist_ok=True)
            shutil.copytree(dd / name, work / 'python' / v / name, symlinks=True)
        say('build', f'uv {v}: default Python {defaults[v]}')
    latest = uvbin[uv_versions[-1]]
    # groups: one resolution per (uv version, Python requests, specs)
    groups = {}
    for key, r in sorted(cands.items()):
        gk = group_key(r)
        g = groups.setdefault(group_id(gk), {'uv_version': r['uv_version'], 'pythons': r['pythons'],
                                             'implicit_python': r['implicit_python'], 'specs': r['specs'], 'tasks': 0})
        g['tasks'] += 1
        r['group'] = group_id(gk)
    for g in groups.values():
        g['targets'] = sorted(set(g['pythons']) | (set(TARGETS_IMPLICIT) | {defaults[g['uv_version']]}
                                                   if g['implicit_python'] else set()), key=lambda p: tuple(map(int, p.split('.'))))
    reused = set()
    if old:
        for gid, g in groups.items():
            og = old['groups'].get(gid)
            v = g['uv_version']
            if og and og.get('offline') and og['uv_version'] == v and old['default_python'].get(v) == defaults[v]:
                g.update({k: x for k, x in og.items() if k != 'tasks'})
                reused.add(gid)
                for w in og['wheels']:
                    if not (tc / 'pool' / w).exists():
                        shutil.copy2(Path(reuse) / 'pool' / w, tc / 'pool' / w)
        say('build', f'reusing {len(reused)} groups of {reuse}')
    say('build', f'{len(groups)} resolution groups, {sum(len(g["targets"]) for gid, g in groups.items() if gid not in reused)} '
                 f'(group, Python) resolutions to run')
    # 1) resolve (uv pip compile: metadata only) per (group, target Python), in parallel
    reqdir = work / 'req'
    reqdir.mkdir(exist_ok=True)

    for gid, g in groups.items():
        (reqdir / f'{gid}.in').write_text('\n'.join(g['specs']) + '\n')

    def resolve(item):
        gid, py = item
        g = groups[gid]
        req = reqdir / f'{gid}.in'
        r = run([uvbin[g['uv_version']], 'pip', 'compile', req, '--python-version', py, '--python-platform',
                 'x86_64-manylinux_2_31', '--no-header', '--no-annotate', '--quiet'], env=env, check=False, timeout=1800,
                retries=4)
        if r.returncode != 0:
            return item, None, r.stderr.strip()[-400:]
        return item, [l.strip() for l in r.stdout.splitlines() if l.strip() and not l.startswith('#')], None

    items = [(gid, py) for gid, g in sorted(groups.items()) if gid not in reused for py in g['targets']]
    with ThreadPoolExecutor(threads) as ex:
        for (gid, py), pins, err in ex.map(resolve, items):
            g = groups[gid]
            if pins is None:
                g.setdefault('unresolvable', {})[py] = err
            elif any(HEAVY.match(p) for p in pins):
                g.setdefault('unresolvable', {})[py] = f'heavy transitive dependency {[p for p in pins if HEAVY.match(p)][:3]}'
            else:
                g.setdefault('pins', {})[py] = pins
    for gid, g in groups.items():
        if gid in reused:
            continue
        ok = sorted(g.get('pins', {}), key=lambda p: tuple(map(int, p.split('.'))))
        v = g['uv_version']
        if g['implicit_python']:
            # an image Python too old for the specs makes uv refine the request to the lowest compatible minor and
            # download it: that Python ships too
            g['refine_python'] = ok[0] if ok and any(tuple(map(int, p.split('.'))) < tuple(map(int, ok[0].split('.')))
                                                    for p in g.get('unresolvable', {})) else None
            g['runtime_pythons'] = sorted({defaults[v]} | ({g['refine_python']} if g['refine_python'] else set()))
        else:
            g['runtime_pythons'] = list(g['pythons'])
        if not ok:
            first = next(iter(g.get('unresolvable', {}).items()), ('-', 'no target Python'))
            g.update(offline=False, reason=f'resolve {first[0]}: {first[1][-300:]}')
    say('build', f"resolved: {sum(1 for g in groups.values() if g.get('offline', True))} groups resolvable for their runtime Pythons")
    # 2) download each unique pin once per Python (once in all for pure-Python wheels), in parallel
    pip_env = {**env, 'UV_PYTHON_INSTALL_DIR': str(work / 'pip-python')}
    pip = [latest, 'tool', 'run', '--python', '3.12', '--from', 'pip==25.2', 'pip']
    run(pip + ['--version'], env=pip_env)   # bootstrap the pip tool env once, before the threads
    by_pin = collections.defaultdict(set)
    files, errors = {}, {}
    for k, name in ((old or {}).get('files') or {}).items():
        py, pin = k.split(' ', 1)
        if (tc / 'pool' / name).exists() or (Path(reuse) / 'pool' / name).exists():
            files[py, pin] = name
    for gid, g in groups.items():
        if g.get('offline') is False or gid in reused:
            continue
        for py, pins in g.get('pins', {}).items():
            for pin in pins:
                if (py, pin) not in files:
                    by_pin[pin].add(py)
                elif not (tc / 'pool' / files[py, pin]).exists():
                    shutil.copy2(Path(reuse) / 'pool' / files[py, pin], tc / 'pool' / files[py, pin])

    def fetch_one(pin, py):
        dl = work / 'dl' / py / re.sub(r'[^A-Za-z0-9_.=-]', '_', pin)
        dl.mkdir(parents=True, exist_ok=True)
        abi = 'cp' + py.replace('.', '')
        cmd = pip + ['download', '--no-deps', '--only-binary=:all:', '--python-version', py, '--implementation', 'cp',
                     '--abi', abi, '--abi', 'abi3', '--abi', 'none', *sum((['--platform', p] for p in PLATFORMS), []),
                     '-d', dl, '--cache-dir', work / 'pip-cache', '--progress-bar', 'off', pin]
        last = None
        for attempt in range(3):   # the proxy resets long transfers now and then
            r = run(cmd, env=pip_env, check=False, timeout=3600)
            got = sorted(f for f in dl.iterdir() if f.suffix == '.whl')
            if r.returncode == 0 and len(got) == 1:
                try:
                    os.link(got[0], tc / 'pool' / got[0].name)
                except FileExistsError:
                    pass
                return got[0].name, None
            last = r.stderr.strip()[-300:] or f'{len(got)} wheel files'
            time.sleep(5 * (attempt + 1))
        return None, last

    def fetch(item):
        pin, pys = item
        pys = sorted(pys, key=lambda p: tuple(map(int, p.split('.'))), reverse=True)
        res = {}
        name, err = fetch_one(pin, pys[0])
        res[pys[0]] = (name, err)
        if name and name.endswith('-none-any.whl'):
            for py in pys[1:]:
                res[py] = (name, None)
        else:
            for py in pys[1:]:
                res[py] = fetch_one(pin, py)
        return pin, res

    say('build', f'{len(by_pin)} unique pins, {sum(len(v) for v in by_pin.values())} (pin, Python) pairs')
    with ThreadPoolExecutor(threads) as ex:
        for n, (pin, res) in enumerate(ex.map(fetch, sorted(by_pin.items())), 1):
            for py, (name, err) in res.items():
                if err:
                    errors[py, pin] = err
                else:
                    files[py, pin] = name
            if n % 25 == 0 or n == len(by_pin):
                say('build', f'downloaded {n}/{len(by_pin)} pins ({len(errors)} failures)')
    for gid, g in sorted(groups.items()):
        if g.get('offline') is False or gid in reused:
            continue
        wheels = set()
        for py, pins in sorted(g['pins'].items()):
            bad = [f'{pin}: {errors[py, pin]}' for pin in pins if (py, pin) in errors]
            if bad:
                g.setdefault('missing_targets', {})[py] = f'download {bad[0][:300]}'
                continue
            wheels |= {files[py, pin] for pin in pins}
        avail = [p for p in g['pins'] if p not in g.get('missing_targets', {})]
        if not avail:
            g.update(offline=False, reason=next(iter(g['missing_targets'].values())))
            continue
        # a runtime Python without its wheels or resolution is not shipped: images that would use it defer to the
        # online script (the offline uvx fails before pytest)
        g['missing_runtime'] = [p for p in g['runtime_pythons'] if p not in avail]
        g['runtime_pythons'] = [p for p in g['runtime_pythons'] if p in avail]
        if g.get('refine_python') not in avail:
            g['refine_python'] = None
        g['wheels'] = sorted(wheels)
        g['wheels_bytes'] = sum((tc / 'pool' / w).stat().st_size for w in wheels)
        if g['wheels_bytes'] > PER_TASK_CAP:
            g.update(offline=False, reason=f"wheels {g['wheels_bytes'] / 2**20:.0f} MB > cap")
    # 3) the CPythons the offline groups need (per uv version), and smoke-only image Pythons
    shipped = collections.defaultdict(set)
    for g in groups.values():
        if g.get('offline') is not False:
            shipped[g['uv_version']] |= set(g['runtime_pythons'])
    for v, minors in sorted(shipped.items()):
        need = sorted(m for m in minors if m not in installed_pythons(work / 'python' / v))
        if need:
            run([uvbin[v], 'python', 'install', *need], retries=8, env={**env, 'UV_PYTHON_INSTALL_DIR': str(work / 'python' / v),
                                                              'UV_PYTHON_BIN_DIR': str(work / f'pybin-{v}')})
        relativize_links(work / 'python' / v)
        say('build', f'uv {v}: ships Pythons {installed_pythons(work / "python" / v)}')
    smoke_py = work / 'smoke-python'
    run([latest, 'python', 'install', *TARGETS_IMPLICIT], retries=8, env={**env, 'UV_PYTHON_INSTALL_DIR': str(smoke_py),
                                                              'UV_PYTHON_BIN_DIR': str(work / 'pybin-smoke')})
    # 4) offline end-to-end checks per group: the group's uvx command with no network, wheels from its find-links only
    #    and only its runtime Pythons, once per image scenario (no Python; each image Python): the interpreter uv picks
    #    and the installed versions must be what an online run gets (the online pins for that Python)
    checks = offline_checks({gid: g for gid, g in groups.items() if gid not in reused}, uvbin, defaults, work, smoke_py,
                            tc / 'pool', threads)
    for gid, problems in checks.items():
        g = groups[gid]
        g['check_failures'] = problems
        if problems and g.get('offline') is not False:
            g.update(offline=False, reason=f'offline check: {problems[0][:250]}')
    online = online_model_check(uvbin, defaults, groups, work, smoke_py)
    for g in groups.values():
        g.setdefault('offline', True)
        g.setdefault('reason', None)
    # 5) task records
    for key, rec in tasks.items():
        if key not in cands:
            rec['offline'] = False
            rec['reason'] = rec.get('reason') or 'no uv bootstrap'
            continue
        g = groups[rec['group']]
        rec['runtime_pythons'] = g['runtime_pythons']
        rec.update(offline=bool(g['offline']), reason=g['reason'])
    # 6) /data layout: one tarball per shipped CPython, the pool, the uv files, manifest.json
    pythons = {}
    for v in uv_versions:
        tars_dir = tc / 'python-tar' / v
        tars_dir.mkdir(parents=True, exist_ok=True)
        for minor, name in sorted(installed_pythons(work / 'python' / v).items()):
            if minor not in shipped.get(v, set()):
                continue
            with tarfile.open(tars_dir / f'{name}.tar', 'w') as tf:
                tf.add(work / 'python' / v / name, arcname=name)
            pythons.setdefault(v, {})[minor] = name
    used = {w for g in groups.values() if g['offline'] for w in g.get('wheels', [])}
    for f in list((tc / 'pool').iterdir()):
        if f.name not in used:
            f.unlink()
    manifest = {
        'version': 1, 'bundle': bundle, 'built_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source': {'tars': [str(t) for t in tars], 'order': str(order), 'limit': limit},
        'uv_versions': uv_versions, 'default_python': defaults, 'pythons': pythons, 'platform': 'x86_64-manylinux_2_31',
        'online_model': online, 'groups': groups, 'tasks': tasks, 'reused_from': str(reuse) if reuse else None,
        'files': {f'{py} {pin}': name for (py, pin), name in sorted(files.items()) if name in used},
        'pool_files': {f.name: wheel_meta(f) for f in sorted((tc / 'pool').iterdir())},
        'uv_files': {str(f.relative_to(tc)): sha256(f) for f in sorted((tc / 'uv').rglob('*')) if f.is_file()},
        'python_tars': {str(f.relative_to(tc)): sha256(f) for f in sorted((tc / 'python-tar').rglob('*.tar'))},
    }
    manifest['bytes'] = {d: sum(f.stat().st_size for f in (tc / d).rglob('*') if f.is_file() and not f.is_symlink())
                         for d in ('uv', 'python-tar', 'pool')}
    manifest['summary'] = summarize(manifest)
    (tc / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    say('build', f'summary {json.dumps(manifest["summary"])}')
    say('build', f'copying to {out}: {sum(1 for _ in tc.rglob("*"))} entries, bytes {manifest["bytes"]}')
    copy_tree(tc, Path(out))
    say('build', 'done')
    return manifest


def offline_checks(groups, uvbin, defaults, work, smoke_py, pool, threads):
    """{group id: [problems]} from the offline end-to-end runs (see build step 4)."""
    pool_meta = {w: wheel_meta(pool / w) for g in groups.values() if g.get('offline') is not False for w in g['wheels']}
    smoke = installed_pythons(smoke_py)
    root = work / 'checks'
    probe = root / 'probe' / 'test_probe.py'
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(PROBE)
    tools = tool_dir(root)
    jobs = []
    for gid, g in sorted(groups.items()):
        if g.get('offline') is False:
            continue
        v = g['uv_version']
        gdir = root / gid
        (gdir / 'wheels').mkdir(parents=True, exist_ok=True)
        for w in g['wheels']:
            if not (gdir / 'wheels' / w).exists():
                os.link(pool / w, gdir / 'wheels' / w)
        write_simple_index(gdir / 'simple', g['wheels'], pool_meta)
        pyd = gdir / 'python'
        pyd.mkdir(exist_ok=True)
        for minor in g['runtime_pythons']:
            name = installed_pythons(work / 'python' / v)[minor]
            if not (pyd / name).exists():
                shutil.copytree(work / 'python' / v / name, pyd / name, symlinks=True, copy_function=os.link)
        avail = {p for p in g['pins'] if p not in g.get('missing_targets', {})}
        scenarios = [None] + ([p for p in TARGETS_IMPLICIT if p in smoke] if g['implicit_python']
                              else [p for p in g['pythons'] if p in smoke])
        for image_py in scenarios:
            if image_py is None:
                expect = g['pythons'][0] if g['pythons'] else defaults[v]
                if expect not in g['runtime_pythons']:
                    continue   # not shipped: the offline run defers to the online script
            elif image_py in avail:
                expect = image_py
            elif image_py in g.get('unresolvable', {}) and g.get('refine_python'):
                expect = g['refine_python']
            else:
                continue   # no compatible Python, or no wheels for it: the offline run defers to the online script
            jobs.append((gid, image_py, expect))

    def one(job):
        gid, image_py, expect = job
        g = groups[gid]
        v = g['uv_version']
        tag = f"{gid}-{image_py or 'none'}"
        tmp = root / 'runs' / tag
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / 'bin').mkdir(parents=True)
        (tmp / 'home').mkdir()
        if image_py:
            exe = smoke_py / smoke[image_py] / 'bin' / f'python{image_py}'
            for n in (f'python{image_py}', 'python3', 'python'):
                os.symlink(exe, tmp / 'bin' / n)
        cmd = [Path(uvbin[v]).with_name('uvx'), *sum((['-p', p] for p in g['pythons']), []),
               *sum((['--with', s] for s in g['specs']), []), 'pytest', '-q', '-s', '-p', 'no:cacheprovider', probe]
        e = {'HOME': str(tmp / 'home'), 'PATH': f"{tmp / 'bin'}:{tools}", 'UV_OFFLINE': '1',
             'UV_DEFAULT_INDEX': (root / gid / 'simple').as_uri(), 'UV_PYTHON_DOWNLOADS': 'never',
             'UV_PYTHON_INSTALL_DIR': str(root / gid / 'python'), 'UV_PYTHON_PREFERENCE': 'system',
             'UV_CACHE_DIR': str(tmp / 'cache'), 'http_proxy': 'http://127.0.0.1:9', 'https_proxy': 'http://127.0.0.1:9'}
        r = run(cmd, env=e, cwd=tmp, check=False, timeout=900)
        m = re.search(r'^PROBE (.*)$', r.stdout, re.M)
        shutil.rmtree(tmp, ignore_errors=True)
        if not m:
            return gid, f'image Python {image_py}: offline uvx failed ({(r.stderr or r.stdout).strip()[-200:]})'
        got = json.loads(m.group(1))
        if '.'.join(got['python'].split('.')[:2]) != expect:
            return gid, f'image Python {image_py}: offline ran Python {got["python"]}, online would run {expect}'
        want = sorted(norm_pin(p) for p in g['pins'][expect])
        have = sorted(norm_pin(d) for d in got['dists'])
        if want != have:
            diff = sorted(set(want) ^ set(have))
            return gid, f'image Python {image_py}: offline installed versions differ {diff[:4]}'
        return gid, None

    out = {gid: [] for gid, g in groups.items() if g.get('offline') is not False}
    with ThreadPoolExecutor(threads) as ex:
        for gid, problem in ex.map(one, jobs):
            if problem:
                out[gid].append(problem)
    say('build', f'offline checks: {len(jobs)} runs, {sum(1 for p in out.values() if p)} groups with failures')
    return out


def online_model_check(uvbin, defaults, groups, work, smoke_py):
    """The interpreter an online run picks, with real downloads into an empty install dir: no image Python -> the uv
    version's default; an image Python too old for pytest 8.4.1 -> the refined minor. Recorded in the manifest."""
    out = {}
    smoke = installed_pythons(smoke_py)
    probe = work / 'checks' / 'probe' / 'test_probe.py'
    for v in uvbin:
        refine = sorted({g['refine_python'] for g in groups.values() if g['uv_version'] == v and g.get('refine_python')})
        for image_py, expect in [(None, defaults[v])] + [('3.8', r) for r in refine[:1]]:
            tmp = work / 'online-model' / f'{v}-{image_py}'
            shutil.rmtree(tmp, ignore_errors=True)
            (tmp / 'bin').mkdir(parents=True)
            if image_py and image_py in smoke:
                for n in (f'python{image_py}', 'python3', 'python'):
                    os.symlink(smoke_py / smoke[image_py] / 'bin' / f'python{image_py}', tmp / 'bin' / n)
            e = base_env(HOME=tmp, PATH=f"{tmp / 'bin'}:{tool_dir(work / 'checks')}", UV_PYTHON_INSTALL_DIR=tmp / 'py',
                         UV_CACHE_DIR=tmp / 'cache')
            r = run([Path(uvbin[v]).with_name('uvx'), '--with', 'pytest==8.4.1', 'pytest', '-q', '-s', '-p',
                     'no:cacheprovider', probe], env=e, cwd=tmp, check=False, timeout=1800)
            m = re.search(r'^PROBE (.*)$', r.stdout, re.M)
            got = json.loads(m.group(1))['python'] if m else None
            shipped = installed_pythons(work / 'python' / v)
            key = f'{v}/{image_py or "none"}'
            out[key] = {'online_python': got, 'expected_minor': expect, 'toolchain_python': shipped.get(expect),
                        'agrees': bool(got and shipped.get(expect, '').startswith(f'cpython-{got}-'))}
            shutil.rmtree(tmp, ignore_errors=True)
            say('build', f'online model {key}: {out[key]}')
    return out


def summarize(m):
    tasks = list(m['tasks'].values())
    why = collections.Counter()
    for r in tasks:
        if not r.get('offline'):
            why[re.sub(r'\s*[\[(].*', '', r.get('reason') or '?')[:80]] += 1
    off = [r for r in tasks if r.get('offline')]
    return {'tasks': len(tasks), 'offline': len(off), 'offline_network_free': sum(1 for r in off if r.get('strip_apt')),
            'offline_guarded_packages': sum(1 for r in off if not r.get('strip_apt')),
            'offline_partial_pythons': sum(1 for r in off if m['groups'][r['group']].get('missing_runtime')),
            'online_reasons': dict(why.most_common()), 'groups': len(m['groups']),
            'groups_offline': sum(1 for g in m['groups'].values() if g.get('offline')),
            'wheels': len(m.get('pool_files', {}))}


def copy_tree(src, dst):
    """local disk -> /data: dd with direct I/O for files over 1 MB (page-cache-bound cp crawls on Lustre), cp
    otherwise; staged as dst.partial and renamed (an existing dst is kept as dst.old-<time>)."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (tmp / rel).mkdir(parents=True, exist_ok=True)
        for name in dirs:
            p = Path(root) / name
            if p.is_symlink():
                os.symlink(os.readlink(p), tmp / rel / name)
        for name in files:
            p, q = Path(root) / name, tmp / rel / name
            if p.is_symlink():
                os.symlink(os.readlink(p), q)
            elif p.stat().st_size > 2 ** 20:
                subprocess.run(['dd', f'if={p}', f'of={q}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
                os.chmod(q, p.stat().st_mode & 0o777)
            else:
                shutil.copy2(p, q)
    if dst.exists():
        dst.rename(dst.with_name(dst.name + time.strftime('.old-%Y%m%dT%H%M%S')))
    tmp.rename(dst)


# ---- worker side: wrapper, offline script, lazy cached staging --------------------------------------------------

# Network-failure signatures of a verifier bootstrap (apt, the astral installer, uv downloads, the cluster proxy). The
# wrapper retries the online script only when pytest did not start and its output shows one of these.
NET_FAIL = (r'proxy|could not resolve|temporary failure|failed to (fetch|download|connect)|connection (refused|reset|'
            r'timed out)|network is unreachable|error sending request|request failed after|uvx?: command not found|'
            r'curl: \([0-9]+\)|operation timed out|unable to (connect|fetch|locate package)|name or service not known|'
            r'tls handshake|ssl_connect')
# tests/test.sh of a wrapped task; the task's own script is test_task.sh. 1) Offline: test_offline.sh when staged.
# 2) Online: test_task.sh, retried (backoff, within half the verifier timeout) while pytest has not started and the
# output shows a network failure. A run whose pytest started is final: tests never run twice.
VERIFIER_WRAPPER = r"""#!/bin/bash
# synth verifier wrapper (synth/verifier_toolchain.py): offline toolchain first (test_offline.sh, when staged), else the
# task's own test script (test_task.sh) online, its network bootstrap retried while pytest has not started; tests that
# started never run twice. Online retries stop at __BUDGET__ s, half the task's verifier timeout.
command -v tee >/dev/null 2>&1 && command -v grep >/dev/null 2>&1 || exec bash /tests/test_task.sh
NET_FAIL='__NET_FAIL__'
log=$(mktemp 2>/dev/null || echo /logs/verifier/.synth_verifier_try.log)
started() {
  [ -f /logs/verifier/ctrf.json ] || grep -Eq \
    "test session starts|^(=+ )?[0-9]+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b|no tests ran" "$log"
}
fresh() { rm -f /logs/verifier/reward.txt /logs/verifier/reward.json; }   # a bootstrap failure's EXIT-trap reward
if [ -f /tests/test_offline.sh ]; then
  echo "[synth verifier] offline toolchain"
  bash /tests/test_offline.sh 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if started; then rm -f "$log"; exit $rc; fi
  echo "[synth verifier] offline: pytest did not start; running the task's own script online"
  fresh
fi
start=$SECONDS; n=0
while true; do
  bash /tests/test_task.sh 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if started || ! grep -Eqi "$NET_FAIL" "$log"; then rm -f "$log"; exit $rc; fi
  n=$((n + 1)); wait_s=$((15 * 2 ** n))
  if [ $n -ge 5 ] || [ $((SECONDS - start + wait_s)) -gt __BUDGET__ ]; then
    echo "[synth verifier] pytest did not start after $n tries; giving up"; rm -f "$log"; exit $rc
  fi
  echo "[synth verifier] pytest did not start (try $n, network failure); retrying in ${wait_s}s"
  fresh
  sleep $wait_s
done
"""
WRAPPER_MARK = '# synth verifier wrapper (synth/verifier_toolchain.py)'


def verifier_timeout(task_dir):
    import tomllib
    try:
        cfg = tomllib.loads((Path(task_dir) / 'task.toml').read_text())
        return float((cfg.get('verifier') or {}).get('timeout_sec') or 900)
    except (OSError, ValueError):
        return 900.0


def wrapper_script(task_dir):
    return VERIFIER_WRAPPER.replace('__BUDGET__', str(int(min(600, verifier_timeout(task_dir) / 2)))) \
        .replace('__NET_FAIL__', NET_FAIL)


def write_atomic(path, text, mode=0o755):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(text)
    tmp.chmod(mode)
    os.replace(tmp, path)


def wrap_verifier(task_dir):
    """Install VERIFIER_WRAPPER as tests/test.sh, the task's script becoming tests/test_task.sh (idempotent).
    Returns True when the task is wrapped."""
    tests = Path(task_dir) / 'tests'
    if (tests / 'test_task.sh').exists():
        if not (tests / 'test.sh').exists() or WRAPPER_MARK not in (tests / 'test.sh').read_text(errors='replace'):
            write_atomic(tests / 'test.sh', wrapper_script(task_dir))   # interrupted earlier between the two steps
        return True
    if not (tests / 'test.sh').exists():
        return False
    script = wrapper_script(task_dir)
    os.link(tests / 'test.sh', tests / 'test_task.sh')
    write_atomic(tests / 'test.sh', script)
    return True


OFFLINE_HEADER = """#!/bin/bash
# synth offline verifier (synth/verifier_toolchain.py): the task's test script with its uv installer line removed{apt},
# run with uv {uv} from /tests/.toolchain, packages from its local index (/tests/.toolchain/simple) and no network
# for uv. Exit 97 = this path does not apply here; the wrapper then runs the task's own script online.
if [ -f /etc/alpine-release ] || {{ command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; }}; then
  echo "[synth verifier] offline: musl libc image (the toolchain is glibc-only); deferring to the online script"; exit 97
fi
if env | grep '^UV_' | grep -qv '^UV_CACHE_DIR='; then
  echo "[synth verifier] offline: the image sets UV_* variables; deferring to the online script"; exit 97
fi
mkdir -p "$HOME/.local/bin"
for f in uv uvx env; do cp -f /tests/.toolchain/uv/$f "$HOME/.local/bin/$f"; done
chmod +x "$HOME/.local/bin/uv" "$HOME/.local/bin/uvx"
export UV_OFFLINE=1 UV_DEFAULT_INDEX=file:///tests/.toolchain/simple UV_PYTHON_DOWNLOADS=never
export UV_PYTHON_INSTALL_DIR=/tests/.toolchain/python UV_PYTHON_PREFERENCE=system
"""
PKG_GUARD = ' || { echo "[synth verifier] offline: package install failed; deferring to the online script"; exit 97; }'


def guard_line(line):
    """`<and-or list> [|| true]` -> `<and-or list> || {echo; exit 97}`: a failed package install (no network) stops
    the offline script before pytest, so the wrapper runs the task's own script online."""
    body = re.sub(r'\s+#[^\'"]*$', '', line.rstrip())
    body = re.sub(r'\s*\|\|\s*(?:true|:)\s*$', '', body)
    return body + PKG_GUARD


def offline_script(text, rec, task_dir=None):
    """test_offline.sh for a task (manifest record; fresh parse of task_dir when given) or None."""
    if not rec or not rec.get('offline') or hashlib.sha256(text.encode()).hexdigest() != rec.get('test_sh_sha256'):
        return None
    if UV_PROJECT.search(text):
        return None
    strip = rec.get('strip_apt') and not (task_dir and apt_sensitive(task_dir))
    lines = text.split('\n')
    lls = logical_lines(text)
    installer_at = next((i for i, (_, _, l) in enumerate(lls) if ASTRAL.search(l)), None)
    if installer_at is None:
        return None
    out = []
    for idx, (a, b, l) in enumerate(lls):
        indent = re.match(r'[ \t]*', l).group(0)
        if ASTRAL.search(l):
            out.append(f'{indent}: # uv installer (offline: uv {rec["uv_version"]} from /tests/.toolchain)')
            continue
        pl = package_line(l)
        if pl:
            if not pl['guardable']:
                return None
            if strip and pl['strippable'] and idx < installer_at:
                out.append(f'{indent}: # offline: dropped {l.strip()[:120]!r} (it only fetched curl for the uv installer)')
            else:
                out.append(indent + guard_line(l.strip()))
            continue
        out.extend(lines[a:b + 1])
    body = '\n'.join(out)
    body = re.sub(r'^#![^\n]*\n', '', body, count=1)
    return OFFLINE_HEADER.format(uv=rec['uv_version'], apt=' and its curl-only package lines' if strip else '') + body


@contextlib.contextmanager
def flocked(path):
    """Exclusive flock on a node-local file (never on /data or /lustre: Lustre has no kernel locks). Serializes
    threads and processes alike (one open file description per acquisition)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


_MANIFESTS = {}
_MANIFEST_LOCK = threading.Lock()


def load_manifest(root):
    """manifest.json of a toolchain root, parsed once per process (reloaded when it changes), or None."""
    path = Path(root) / 'manifest.json'
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    with _MANIFEST_LOCK:
        hit = _MANIFESTS.get(str(path))
        if hit and hit[0] == key:
            return hit[1]
        try:
            m = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        _MANIFESTS[str(path)] = (key, m)
        return m


def _copy_verified(src, dst, digest):
    """src (on /data) -> dst (node-local) through a private temp name, sha256-checked, atomically renamed."""
    tmp = dst.with_name(f'.{dst.name}.{os.getpid()}.{threading.get_ident()}.partial')
    h = hashlib.sha256()
    with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
        for chunk in iter(lambda: fi.read(4 << 20), b''):
            h.update(chunk)
            fo.write(chunk)
    if digest and h.hexdigest() != digest:
        tmp.unlink(missing_ok=True)
        raise OSError(f'sha256 mismatch for {src}')
    shutil.copymode(src, tmp)
    os.replace(tmp, dst)


def ensure_local(manifest, root, local, rel):
    """Copy one toolchain piece from root (/data) into the node-local cache once: 'uv/<v>/<f>', 'pool/<wheel>' or
    'python/<v>/<install dir>' (from python-tar/<v>/<dir>.tar). Safe for concurrent callers (per-piece flock)."""
    root, local = Path(root), Path(local)
    dst = local / rel
    if dst.exists():
        return dst
    with flocked(local / '.locks' / (hashlib.sha1(rel.encode()).hexdigest() + '.lock')):
        if dst.exists():
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.startswith('python/'):
            _, v, name = rel.split('/')
            tar_rel = f'python-tar/{v}/{name}.tar'
            tmp_tar = local / '.tmp' / f'{name}.{os.getpid()}.{threading.get_ident()}.tar'
            tmp_tar.parent.mkdir(parents=True, exist_ok=True)
            _copy_verified(root / tar_rel, tmp_tar, manifest['python_tars'].get(tar_rel))
            tmp_dir = local / '.tmp' / f'{name}.{os.getpid()}.{threading.get_ident()}'
            shutil.rmtree(tmp_dir, ignore_errors=True)
            with tarfile.open(tmp_tar) as tf:
                tf.extractall(tmp_dir, filter='tar')
            tmp_tar.unlink()
            os.rename(tmp_dir / name, dst)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif rel.startswith('pool/'):
            _copy_verified(root / rel, dst, manifest['pool_files'][rel.split('/', 1)[1]]['sha256'])
        else:
            _copy_verified(root / rel, dst, manifest['uv_files'].get(rel))
    return dst


def _link_tree(src, dst):
    """Hardlink a file tree (copy when the cache is on another filesystem)."""
    def link(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    shutil.copytree(src, dst, symlinks=True, copy_function=link)


def stage_task_cached(task_dir, rec, manifest, toolchain_root, local_cache, script):
    """Stage tests/.toolchain (uv files, the task's runtime Pythons, its wheels and their simple index) and
    tests/test_offline.sh for one task, from the node-local cache (filled from toolchain_root once per piece).
    Idempotent: a staging with the same stamp is kept. Call with the task's lock held. Returns True when it
    (re)staged now, False when an identical staging was already there."""
    task_dir, local = Path(task_dir), Path(local_cache)
    tests = task_dir / 'tests'
    group = manifest['groups'][rec['group']]
    v = rec['uv_version']
    stamp = json.dumps({'built_at': manifest['built_at'], 'group': rec['group'],
                        'script': hashlib.sha256(script.encode()).hexdigest()})
    dst = tests / '.toolchain'
    try:
        if (dst / '.stamp').read_text() == stamp and (tests / 'test_offline.sh').read_text() == script:
            return False
    except OSError:
        pass
    pieces = [f'uv/{v}/{f}' for f in ('uv', 'uvx', 'env')] + [f'pool/{w}' for w in group['wheels']] + \
        [f'python/{v}/{manifest["pythons"][v][p]}' for p in group['runtime_pythons']]
    for rel in pieces:
        ensure_local(manifest, toolchain_root, local, rel)
    tmp = local / '.staging' / f'{task_dir.name}.{os.getpid()}.{threading.get_ident()}'
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / 'uv').mkdir(parents=True)
    (tmp / 'wheels').mkdir()
    (tmp / 'python').mkdir()
    for rel in pieces:
        kind, rest = rel.split('/', 1)
        if kind == 'uv':
            _link_tree_file(local / rel, tmp / 'uv' / Path(rel).name)
        elif kind == 'pool':
            _link_tree_file(local / rel, tmp / 'wheels' / rest)
        else:
            _link_tree(local / rel, tmp / 'python' / Path(rel).name)
    write_simple_index(tmp / 'simple', group['wheels'], manifest['pool_files'])
    (tmp / '.stamp').write_text(stamp)
    (tests / 'test_offline.sh').unlink(missing_ok=True)   # never an offline script without its toolchain
    if dst.exists():
        old = local / '.staging' / f'{task_dir.name}.old.{os.getpid()}.{threading.get_ident()}'
        os.rename(dst, old)
        os.rename(tmp, dst)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.rename(tmp, dst)
    write_atomic(tests / 'test_offline.sh', script)
    return True


def _link_tree_file(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def unstage(task_dir):
    tests = Path(task_dir) / 'tests'
    (tests / 'test_offline.sh').unlink(missing_ok=True)
    shutil.rmtree(tests / '.toolchain', ignore_errors=True)


def prepare_verifier(task_dir, toolchain_root=TOOLCHAIN, local_cache=None, bundle=BUNDLE, holder=None):
    """Prepare one task's verifier right before an attempt (worker, per attempt; idempotent and safe for concurrent
    attempts of the same task on one node). Installs the wrapper as tests/test.sh for tasks with a pinned uv
    bootstrap, and stages the offline toolchain when the manifest and a fresh parse of the task agree it is
    equivalent. holder (e.g. the attempt id) registers the attempt so release_verifier() can unstage the toolchain
    when the last attempt of the task on this node is done. local_cache: node-local directory on the same
    filesystem as task_dir (hardlinks), e.g. $_CONDOR_SCRATCH_DIR/verifier-toolchain. Returns
    {'task', 'mode': 'offline'|'online'|'unchanged', 'reason', 'wrapped', 'staged_now', 'strip_apt', 'seconds'}."""
    t0 = time.time()
    task_dir = Path(task_dir).resolve()
    local = Path(local_cache or os.path.join(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir(),
                                             'verifier-toolchain'))
    out = {'task': task_dir.name, 'mode': 'online', 'reason': None, 'wrapped': False, 'staged_now': False,
           'strip_apt': None}
    with flocked(local / '.locks' / f'task-{task_dir.name}.lock'):
        tests = task_dir / 'tests'
        fresh = parse(task_dir)
        if fresh['kind'] == 'none':
            out.update(mode='unchanged', reason=fresh['reason'])
        else:
            out['wrapped'] = wrap_verifier(task_dir)
            manifest = load_manifest(toolchain_root) if toolchain_root else None
            rec = (manifest or {}).get('tasks', {}).get(f'{bundle}/{task_dir.name}')
            text = (tests / 'test_task.sh').read_text(errors='replace')
            if manifest is None:
                out['reason'] = f'no toolchain manifest at {toolchain_root}'
            elif rec is None:
                out['reason'] = 'task not in the toolchain manifest'
            elif not rec.get('offline'):
                out['reason'] = rec.get('reason') or 'not offline in the manifest'
            elif fresh.get('reason'):
                out['reason'] = f"task changed since the build: {fresh['reason']}"
            elif fresh.get('test_sh_sha256') != rec.get('test_sh_sha256'):
                out['reason'] = 'test.sh changed since the build'
            elif group_key(fresh) != group_key(rec):
                out['reason'] = 'uv command changed since the build'
            else:
                rec = {**rec, 'strip_apt': bool(rec.get('strip_apt') and fresh.get('strip_apt'))}
                script = offline_script(text, rec, task_dir)
                if script is None:
                    out['reason'] = 'no offline script'
                else:
                    try:
                        out['staged_now'] = stage_task_cached(task_dir, rec, manifest, toolchain_root, local, script)
                        out.update(mode='offline', strip_apt=rec['strip_apt'])
                    except (OSError, KeyError, tarfile.TarError) as e:
                        unstage(task_dir)
                        out['reason'] = f'staging failed: {e!r}'[:300]
            if out['mode'] != 'offline':
                unstage(task_dir)
            if holder and out['mode'] == 'offline':
                (local / '.holders' / task_dir.name).mkdir(parents=True, exist_ok=True)
                (local / '.holders' / task_dir.name / str(holder)).touch()
    out['seconds'] = round(time.time() - t0, 3)
    return out


def release_verifier(task_dir, local_cache=None, holder=None):
    """After an attempt: drop holder's registration; when no attempt of the task on this node holds it any more,
    remove tests/.toolchain and tests/test_offline.sh (the wrapper stays; the next prepare_verifier re-stages from
    the node-local cache). Returns True when it unstaged."""
    task_dir = Path(task_dir).resolve()
    local = Path(local_cache or os.path.join(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir(),
                                             'verifier-toolchain'))
    with flocked(local / '.locks' / f'task-{task_dir.name}.lock'):
        holders = local / '.holders' / task_dir.name
        if holder:
            (holders / str(holder)).unlink(missing_ok=True)
        if holders.exists() and any(holders.iterdir()):
            return False
        unstage(task_dir)
        return True


# ---- CLI -----------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('survey', 'build'):
        s = sub.add_parser(name)
        s.add_argument('--tasks-tar', action='append', required=True, help='task tar (members tasks/<id>/...); repeatable')
        s.add_argument('--order', required=True)
        s.add_argument('--limit', type=int, default=0, help='first N tasks of --order (0: all)')
        s.add_argument('--bundle', default=BUNDLE)
        s.add_argument('--jobs', type=int, default=os.cpu_count() or 8)
        s.add_argument('--work', default=os.environ.get('_CONDOR_SCRATCH_DIR', tempfile.gettempdir()) + '/vtc')
    b = sub.choices['build']
    b.add_argument('--out', default=str(TOOLCHAIN))
    b.add_argument('--threads', type=int, default=16)
    b.add_argument('--reuse', help='earlier toolchain root: reuse its groups and downloads (incremental extension)')
    a = ap.parse_args()
    if a.cmd == 'survey':
        work = Path(a.work)
        names = task_names(a.order, a.limit)
        found = extract_tasks(a.tasks_tar, names, work)
        tasks = survey(work / 'tasks', [n for n in names if n in found], a.bundle, a.jobs)
        recs = list(tasks.values())
        print(json.dumps({'tasks': len(recs), 'candidates': sum(1 for r in recs if r['kind'] == 'uv' and not r.get('reason')),
                          'reasons': collections.Counter(re.sub(r'\s*[\[(].*', '', r.get('reason') or '-') for r in recs)},
                         indent=1))
        return
    if a.cmd == 'build':
        m = build(a.out, a.tasks_tar, a.order, a.limit, a.work, a.bundle, a.jobs, a.threads, a.reuse)
        print(json.dumps(m['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
