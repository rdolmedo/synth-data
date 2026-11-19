"""Agentic authoring (issue #20 follow-up, 2026-11-01): the generator explores the real asset in a sandbox before it
writes the task, and may keep using the sandbox while it repairs the task after checks.

The sandbox is the task base image with the asset's files under /tasksynth/asset/ (a writable copy), package mirrors
reachable through the cluster proxy. The generator runs a command through a bash tool (OpenAI tools API; the server parses its native tool calls); the
round driver executes it and returns exit code and output. Nothing done in the sandbox reaches the task build: the
task is built from the generator's files and setup script as before.

Budgets (user, 2026-11-01: "we don't need to be stingy with interaction time ... we want good tasks"): bounded by the
generator's 132k-token context, not by turn count. Reasoning of earlier turns is not resent, so the budget is the
tool output kept in the conversation.
"""
import asyncio
import json
import os
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from synth.common import PROXY_ENV

from .check import _IMAGE_USERS, release_image
from .package import BASE_REF

BASH_TOOL = {'type': 'function', 'function': {
    'name': 'bash',
    'description': 'Run one shell command in the authoring sandbox (cwd /tasksynth/asset) and return its exit code and '
                   'output (long output is cut in the middle). Up to 10 minutes per command.',
    'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'the shell command'}},
                   'required': ['command']}}}
MAX_COMMANDS = 80
CMD_TIMEOUT = 600
OUTPUT_CHARS = 8000          # per command, head and tail
TOTAL_OUTPUT_CHARS = 600_000  # a ceiling only: the context limits below govern
# Context (ag1, 2026-11-01: after 65 commands the conversation hit the 132k window and the package reply had no room;
# the author's own long commands, not tool output, filled it). The final reply needs ~50-80k tokens (reasoning +
# package), so: trim older turns once the prompt passes COMPACT_AT, and withdraw the tool past STOP_AT.
COMPACT_AT = 40_000      # prompt tokens
STOP_AT = 55_000
KEEP_LAST = 8            # tool results (and their calls) kept whole
RENDER_CHARS = 12_000    # the asset rendering in an agentic prompt: the author can open the files itself
# --workflow lean (iteration 5, 2026-11-02): compaction trims every tool result before the last KEEP_LAST each time the
# prompt passes COMPACT_AT, so near the threshold it rewrites the middle of the prompt on every step and the server
# re-prefills ~20k tokens per call (gen-ds2 prefix hit rate 65% in production). Lean compaction waits for a higher
# mark and then cuts deep, so the prompt is append-only (cached) for many steps between compactions.
LEAN_COMPACT_AT = 48_000
LEAN2_NUDGE_AT = (0.5, 0.3, 0.15)
# lean2 (2026-11-02, user: why compact at 40k?): compaction rewrites the prompt and costs a full re-prefill, and the
# 55k stop dated from iteration 1's 130k overflow. lean2 sessions (~35 steps, 4k-char outputs) mostly stay well below
# the 132k context, so compact only near it (once, deep) and stop just short of it: the prompt stays append-only.
LEAN2_COMPACT_AT = 100_000
LEAN2_STOP_AT = 120_000
LEAN_KEEP_LAST = 2
LEAN_OUTPUT_CHARS = 4000
SANDBOX_MEMORY_MB = 8192  # per author sandbox (PSS guard kills its largest process): ws1 (48 authors) went over
                          # the round job's 96 GiB on 2026-11-01 and was held


def command_of(call):
    """A bash tool call -> its command, else None. (A text protocol with <bash> blocks failed: DeepSeek closes the
    block with its native DSML tool-call tokens, ag0 2026-11-01; the server's tool parser handles those.)"""
    if call.get('name') != 'bash':
        return None
    try:
        args = json.loads(call.get('arguments') or '{}')
    except ValueError:
        return None
    cmd = args.get('command') if isinstance(args, dict) else None
    return cmd.strip() if isinstance(cmd, str) and cmd.strip() else None


def clip_output(text, n=OUTPUT_CHARS):
    if len(text) <= n:
        return text
    return f'{text[:n // 2]}\n[... {len(text) - n} characters cut ...]\n{text[-n // 2:]}'


FAST_START = True
_BASE = {}
_BASE_LOCK = []


async def author_base(store):
    """The shared author image (task base image, empty /tasksynth/asset), built once per process and never released."""
    if not _BASE_LOCK:
        _BASE_LOCK.append(asyncio.Lock())
    async with _BASE_LOCK[0]:
        if id(store) not in _BASE:
            ctx = Path(tempfile.mkdtemp(prefix='explore-base-'))
            (ctx / 'Dockerfile').write_text(f'FROM {BASE_REF}\nRUN mkdir -p /tasksynth/asset /tasksynth/.src\n'
                                            'WORKDIR /tasksynth/asset\n')
            key = store.context_key(ctx, 'Dockerfile')
            _IMAGE_USERS[key] = _IMAGE_USERS.get(key, 0) + 1
            # pinned for the process's lifetime: every author sandbox runs on it (harbor_ns eviction, #451 lesson 7)
            _BASE[id(store)] = await store.build_pinned(ctx, label='explore-base', token=f'explore-base-{os.getpid()}-{secrets.token_hex(4)}')
        return _BASE[id(store)]


class Explorer:
    """Async context manager: a running sandbox with the asset; run(cmd) -> text for the generator."""

    def __init__(self, store, asset_tar, label, live=None, max_commands=MAX_COMMANDS, package_cache=None,
                 fetch_log=None):
        """live: a file that gets one JSON line per command as it runs (to watch exploration in progress);
        package_cache: None, 'apt' or 'apt+pip' -- the sandbox's apt (all its plain HTTP) and, with 'apt+pip', its pip
        go through the job's package cache (synth.package_cache) via a package proxy in the sandbox, which falls back
        to the cluster proxy while the cache is down; fetch_log: where that proxy's request log is copied at the end."""
        self.store, self.asset_tar, self.label, self.live = store, Path(asset_tar), label, live
        self.package_cache, self.fetch_log, self.package_proxy = package_cache, fetch_log, None
        self.proxy_env = PROXY_ENV   # the sandbox's proxy variables (the cluster proxy)
        self.cache_registry = None   # () -> the cache's registry or None while down; default: this job's cache
        self.commands, self.output_chars, self.log, self.last_prompt = 0, 0, [], 0
        self.max_commands, self.checks = max_commands, 0
        self.completion, self.token_budget = 0, TOKEN_BUDGET
        self.output_cap = OUTPUT_CHARS
        self.nudge_at = NUDGE_AT
        self.grace = False
        self.nudged = set()
        self.box = self.key = self.ctx = self.pin = None

    async def __aenter__(self):
        try:
            return await self._enter()
        except BaseException:   # a failed start (asset unpack, ...) leaves no sandbox, relay or image behind
            await self.__aexit__(None, None, None)
            raise

    async def _enter(self):
        from harbor_ns.sandbox import Sandbox
        self.ctx = Path(tempfile.mkdtemp(prefix='explore-'))
        shutil.copyfile(self.asset_tar, self.ctx / 'asset.tar.zst')
        if FAST_START:
            # no image build per author: the shared base image (built once per process) plus the asset unpacked in the
            # box from a bind of this scratch copy (2026-11-02: 1,800 authors at 128 parallel ~2-min builds took ~30 min
            # to start; the servers sat at ~5% load meanwhile)
            self.key = None
            _, rootfs, imeta = await author_base(self.store)
            self.box = Sandbox(cfg=self.store.rt, lower=[rootfs], hostname='author', binds=[(str(self.ctx), '/tasksynth/.src')],
                               base_env={**imeta.env, 'HOME': '/root', **self.proxy_env}, internet=True,
                               memory_limit_mb=SANDBOX_MEMORY_MB)
            await self.box.start()
            await self.use_package_cache()
            r = await self.box.exec('mkdir -p /tasksynth/asset && tar --zstd -xf /tasksynth/.src/asset.tar.zst -C /tasksynth/asset',
                                    timeout=900)
            if r.return_code != 0:
                raise RuntimeError(f'asset unpack failed (rc {r.return_code}): {(r.stderr or r.stdout or "")[-400:]}')
            return self
        (self.ctx / 'Dockerfile').write_text(
            f'FROM {BASE_REF}\nCOPY asset.tar.zst /tmp/asset.tar.zst\n'
            'RUN mkdir -p /tasksynth/asset && tar --zstd -xf /tmp/asset.tar.zst -C /tasksynth/asset '
            '&& rm /tmp/asset.tar.zst\nWORKDIR /tasksynth/asset\n')
        self.key = self.store.context_key(self.ctx, 'Dockerfile')
        _IMAGE_USERS[self.key] = _IMAGE_USERS.get(self.key, 0) + 1
        self.pin = f'explore-{os.getpid()}-{secrets.token_hex(4)}'   # pinned while the author sandbox runs on it
        self.key, rootfs, imeta = await self.store.build_pinned(self.ctx, label=f'explore-{self.label}', token=self.pin)
        self.box = Sandbox(cfg=self.store.rt, lower=[rootfs], hostname='author',
                           base_env={**imeta.env, 'HOME': '/root', **self.proxy_env}, internet=True,
                           memory_limit_mb=SANDBOX_MEMORY_MB)
        await self.box.start()
        await self.use_package_cache()
        return self

    async def use_package_cache(self):
        """Point the sandbox's plain-HTTP proxy (and with 'apt+pip' its pip index) at a package proxy that uses the
        job's package cache. HTTPS keeps the cluster proxy. Any failure leaves the sandbox on the cluster proxy."""
        from harbor_ns.sandbox import PIP_CACHE_TRUST, package_cache
        registry = self.cache_registry or package_cache
        if not self.package_cache or not registry():
            return
        from synth.package_proxy import PIP_INDEX, SANDBOX_URL, attach
        pip = self.package_cache == 'apt+pip'
        try:
            self.package_proxy = await attach(self.box, self.ctx / 'package-proxy', self.proxy_env['http_proxy'], allow=None,
                                              cache=registry, pip=pip, tag=f'author-{self.label}'[:100])
        except Exception as e:  # noqa: BLE001
            print(f'[explore] {self.label}: package proxy not started, the cluster proxy stays: {e}', flush=True)
            return
        await self.box.update_env({'http_proxy': SANDBOX_URL, 'HTTP_PROXY': SANDBOX_URL,
                                   **({'PIP_INDEX_URL': PIP_INDEX, **PIP_CACHE_TRUST} if pip else {})})

    async def __aexit__(self, *exc):
        try:
            if self.package_proxy is not None:
                proxy, relay = self.package_proxy
                self.package_proxy = None
                await relay.close()
                await proxy.close()
                if self.fetch_log is not None:
                    await asyncio.get_running_loop().run_in_executor(None, self.keep_fetch_log)
            if self.box is not None:
                await self.box.stop()
        finally:
            if self.key is not None:
                if self.pin:
                    self.store.unmark_in_use(self.store.image_dir(self.key), self.pin)
                release_image(self.store, self.key)
            if self.ctx is not None:
                shutil.rmtree(self.ctx, ignore_errors=True)

    def keep_fetch_log(self):
        try:
            shutil.copyfile(self.ctx / 'package-proxy' / 'requests.jsonl', self.fetch_log)
        except OSError:
            pass

    def left(self):
        return self.max_commands - self.commands, TOTAL_OUTPUT_CHARS - self.output_chars

    def spent(self):
        """Commands, output or generated tokens used up."""
        return min(self.left()) <= 0 or self.completion >= self.token_budget

    async def run(self, cmd):
        self.commands += 1
        t0 = time.time()
        r = await self.box.exec(cmd, cwd='/tasksynth/asset', timeout=CMD_TIMEOUT)
        out = (r.stdout or '') + (f'\n[stderr]\n{r.stderr}' if r.stderr else '')
        kept = clip_output(out, self.output_cap)
        self.output_chars += len(kept)
        n_left, chars_left = self.left()
        tok_left = max(0, self.token_budget - self.completion)
        share = min(n_left / max(1, self.max_commands), tok_left / max(1, self.token_budget))
        due = [x for x in self.nudge_at if share <= x and x not in self.nudged]
        if getattr(self, 'workspace', False) and due:
            self.nudged.update(due)
            t = await self.box.exec(f'test -f {WORKSPACE}/task.json', timeout=30)
            if t.return_code != 0:
                kept += (f'\n[{n_left} commands and ~{tok_left // 1000}k tokens of your own output left, and '
                         f'{WORKSPACE}/task.json does not exist yet: write the task files under {WORKSPACE} now, check '
                         f'them with check_task, and submit]')
        status = f'exit code {r.return_code}' + (' (timed out after 600 s)' if r.timed_out else '')
        self.log.append({'cmd': cmd[:2000], 'rc': r.return_code, 'out_chars': len(out), 'secs': round(time.time() - t0, 1)})
        if self.live is not None:
            with open(self.live, 'a') as f:
                f.write(json.dumps({**self.log[-1], 'time': time.time(), 'out_head': kept[:400]}) + '\n')
        note = (f'[{status}; {n_left} commands, ~{tok_left // 1000}k tokens of your own output (thinking included) and '
                f'~{chars_left // 4000}k tokens of command output left]'
                if n_left > 0 and chars_left > 0 else
                ('[command budget used up: finish /task with what you have and call submit]'
                 if getattr(self, 'workspace', False) else
                 '[sandbox budget used up: write the complete task now, in the output format]'))
        return f'{note}\n{kept}' if kept.strip() else f'{note}\n(no output)'


def _trim(text, n):
    return text if len(text) <= n else f'{text[:n * 2 // 3]}\n[... {len(text) - n} characters trimmed to save context ...]\n{text[-n // 3:]}'


def compact(messages, keep_last=KEEP_LAST):
    """Shrink older turns in place: tool results and long commands before the last `keep_last` tool results, and
    every assistant text but the latest (earlier task drafts). The system and first user message stay whole."""
    tool_idx = [i for i, m in enumerate(messages) if m.get('role') == 'tool']
    cut = tool_idx[-keep_last] if len(tool_idx) >= keep_last else (tool_idx[0] if tool_idx else len(messages))
    last_text = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant' and len(m.get('content') or '') > 2000),
                    default=-1)
    for i, m in enumerate(messages[2:], start=2):
        if m.get('role') == 'tool' and i < cut:
            m['content'] = _trim(m['content'], 600)
        elif m.get('role') == 'assistant':
            if m.get('content') and i != last_text and len(m['content']) > 2000:
                m['content'] = _trim(m['content'], 1500)
            for tc in m.get('tool_calls') or []:
                args = tc['function']['arguments']
                if i < cut and len(args) > 1500:
                    try:
                        cmd = json.loads(args).get('command', '')
                    except ValueError:
                        cmd = args
                    tc['function']['arguments'] = json.dumps({'command': _trim(cmd, 1200)})


# ---------------------------------------------------------------- workspace authoring (iteration 2)
# ag1/ag2 (2026-11-01): the author tested in the sandbox but retyped the whole package from memory into one reply, and
# the two drifted (judge fails were mostly "consistency": sketch values or docs not matching the built files). In
# workspace mode the author builds the task as files in /task, can run the round's real build + checks on it
# (check_task), and submits it; the package is exported from /task, so what was tested is what ships.
WORKSPACE = '/task'
WORKSPACE_MAX_COMMANDS = 200   # ws1/ws2: every failure had spent 150 commands without finishing /task
REPAIR_COMMANDS = 40           # granted per failed submission (round: a fifth of --author-commands)
REPLY_TOKENS = 32_768          # per generator call while it has tools (reasoning + call): bounds a runaway reply; a
                               # step averages ~2.4k tokens
TOKEN_BUDGET = 800_000         # completion tokens per author, then only submit is left (2026-11-01: ws5 median 456k,
                               # max 633k; ag1/ws2 without scope reached ~1M)
NUDGE_AT = (0.3, 0.15, 0.05)   # share of commands or tokens left: remind the author to write /task if task.json is
                               # still missing (60/30/10 of 200 commands)
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.git'}
SKIP_SUFFIXES = ('.pyc', '.pyo')
MAX_CHECKS = 8
WORKSPACE_MAX_BYTES = 20_000_000
CHECK_TOOL = {'type': 'function', 'function': {
    'name': 'check_task',
    'description': 'Build the task in /task exactly as the pipeline will (a fresh image from your files and setup.sh, '
                   'the asset under /tasksynth/asset/ during setup) and run the automated checks. Returns the problems '
                   'found, or that the checks pass.',
    'parameters': {'type': 'object', 'properties': {}}}}
SUBMIT_TOOL = {'type': 'function', 'function': {
    'name': 'submit',
    'description': 'Submit the task in /task as finished. It is built, checked and reviewed; if it fails, you get the '
                   'problems back and can fix /task and submit again.',
    'parameters': {'type': 'object', 'properties': {}}}}


async def export_package(ex):
    """/task in the sandbox -> package dict (package.parse's shape). Raises PackageError with a message for the author."""
    import base64
    import io
    import tarfile
    from .package import PackageError, validate
    r = await ex.box.exec(f'test -d {WORKSPACE} && du -sb {WORKSPACE} | cut -f1', timeout=120)
    if r.return_code != 0 or not r.stdout.strip().isdigit():
        raise PackageError(f'{WORKSPACE} does not exist: write the task there (see the layout in the request)')
    if int(r.stdout.strip()) > WORKSPACE_MAX_BYTES:
        raise PackageError(f'{WORKSPACE} holds {int(r.stdout.strip())} bytes; keep it to the task files (under '
                           f'{WORKSPACE_MAX_BYTES // 1_000_000} MB): data and binaries are produced by setup.sh')
    r = await ex.box.exec(f'cd {WORKSPACE} && tar czf - . | base64 -w0', timeout=300)
    if r.return_code != 0:
        raise PackageError(f'could not read {WORKSPACE}: {r.stderr[-500:]}')
    data = base64.b64decode(r.stdout)
    root = Path(tempfile.mkdtemp(prefix='workspace-'))
    try:
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(root, filter='data')

        def text(rel, required=True):
            p = root / rel
            if not p.is_file():
                if required:
                    raise PackageError(f'{WORKSPACE}/{rel} is missing')
                return None
            try:
                return p.read_text()
            except UnicodeDecodeError:
                raise PackageError(f'{WORKSPACE}/{rel} is not UTF-8 text') from None
        try:
            meta = json.loads(text('task.json'))
        except ValueError as e:
            raise PackageError(f'{WORKSPACE}/task.json is not valid JSON: {e}') from None
        files = {}
        fdir = root / 'files'
        if fdir.is_dir():
            for p in sorted(fdir.rglob('*')):
                if p.is_file():
                    if SKIP_PARTS & set(p.relative_to(fdir).parts) or p.suffix in SKIP_SUFFIXES or p.name == '.DS_Store':
                        continue   # build byproducts the author's tests left behind
                    rel = '/' + str(p.relative_to(fdir))
                    try:
                        body = p.read_text()
                    except UnicodeDecodeError:
                        raise PackageError(f'{WORKSPACE}/files{rel} is binary: files/ holds text only; produce binaries, '
                                           'archives, databases and media from setup.sh') from None
                    files[rel] = body if body.endswith('\n') else body + '\n'
        setup = text('setup.sh', required=False) or '#!/bin/bash\nset -euo pipefail\n'
        pkg = {'meta': meta, 'instruction': text('instruction.md').strip() + '\n', 'files': files, 'setup': setup,
               'services': text('services.sh', required=False),
               'sketch': (text('solution_sketch.md', required=False) or '').strip() + '\n'}
        validate(pkg)
        return pkg
    finally:
        shutil.rmtree(root, ignore_errors=True)
