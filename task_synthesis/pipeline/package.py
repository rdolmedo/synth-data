"""Parse the generator's reply into a task package and write it out as a Harbor task directory.

Layout of a materialized task:
  task.toml, instruction.md, tasksynth.json (spec, package metadata, provenance)
  environment/Dockerfile            FROM the shared base + thin layer (extra packages, files, setup, services)
  environment/files/<abs path>      the author's files
  environment/tasksynth/            setup.sh, services.sh, entrypoint (when there are services or setup writes
                                    /etc/hosts, which the container runtime replaces at start)
  solution/sketch.md                the author's private sketch (never shown to the agent)
  tests/test.sh                     no verifier (BLUEPRINT.md, decision 1): reward 0, and final_state.py records the
                                    declared outputs and a file tree for the trace judge (tests/paths.json)
"""
import json
import re
import shutil
from pathlib import Path

BASE_REF = 'local/tasksynth-base:2'
PKG = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.+_:=<>~,\-\[\]]*$')
SLUG = re.compile(r'^[a-z0-9][a-z0-9-]{2,60}$')
FORBIDDEN = ('/proc', '/sys', '/dev', '/tasksynth-meta', '/logs', '/tests', '/solution', '/oracle')
MAX_FILES, MAX_BYTES = 80, 1_500_000


class PackageError(ValueError):
    pass


def _block(text, tag, required=True):
    m = re.search(rf'<{tag}>\s*\n?(.*?)\n?\s*</{tag}>', text, re.S)
    if not m:
        if required:
            raise PackageError(f'missing <{tag}> block')
        return None
    return m.group(1)


def _task_json(text):
    """The task_json block; a block left unclosed ends where the next block starts (r3: long defect_check
    commands made the generator drop </task_json>)."""
    raw = _block(text, 'task_json', required=False)
    if raw is not None:
        return raw
    m = re.search(r'<task_json>\s*\n?(.*?)(?=^<(?:instruction|file |setup|services|solution_sketch)\b)', text, re.S | re.M)
    if m:
        return m.group(1)
    if '<task_json>' in text:
        raise PackageError('the <task_json> block is not closed with </task_json>')
    raise PackageError('missing <task_json> block')


def parse(text):
    """Reply text -> package dict. Raises PackageError with a message the generator can act on."""
    raw = _task_json(text)
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PackageError(f'<task_json> is not valid JSON: {e}') from None
    files = {}
    # The closing tag must start a line, so file bodies may mention </file> mid-line.
    for m in re.finditer(r'<file path="([^"]+)">[ \t]*\n(.*?)^</file>[ \t]*$', text, re.S | re.M):
        path, body = m.group(1).strip(), m.group(2)
        files[path] = body if body.endswith('\n') else body + '\n'
    pkg = {'meta': meta, 'instruction': _block(text, 'instruction').strip() + '\n', 'files': files,
           'setup': _block(text, 'setup', required=False) or '#!/bin/bash\nset -euo pipefail\n',
           'services': _block(text, 'services', required=False),
           'sketch': (_block(text, 'solution_sketch', required=False) or '').strip() + '\n'}
    validate(pkg)
    return pkg


def validate(pkg):
    meta, problems = pkg['meta'], []
    for key in ('slug', 'title', 'kind', 'inputs', 'outputs', 'summary'):
        if key not in meta:
            problems.append(f'task_json lacks "{key}"')
    if meta.get('slug') and not SLUG.match(str(meta['slug'])):
        problems.append(f'slug {meta.get("slug")!r} is not kebab-case (3-61 chars)')
    if meta.get('kind') not in ('none', 'scaffold', 'puzzle', 'substrate'):
        problems.append(f'kind {meta.get("kind")!r} is not one of none/scaffold/puzzle/substrate')
    for key in ('apt', 'pip'):
        bad = [p for p in meta.get(key) or [] if not isinstance(p, str) or not PKG.match(p)]
        if bad:
            problems.append(f'{key} has invalid package names: {bad[:5]}')
    paths = list(meta.get('inputs') or []) + list(meta.get('outputs') or []) + list(pkg['files'])
    for p in paths:
        if not isinstance(p, str) or not p.startswith('/') or '..' in p.split('/'):
            problems.append(f'path {p!r} is not a clean absolute path')
        elif p.startswith(FORBIDDEN):
            problems.append(f'path {p} is in a reserved location')
    overlap = set(meta.get('inputs') or []) & set(meta.get('outputs') or [])
    if overlap:
        problems.append(f'paths are both inputs and outputs: {sorted(overlap)[:5]}')
    if meta.get('services') and not pkg['services']:
        problems.append('"services" is true but there is no <services> block')
    if len(pkg['files']) > MAX_FILES:
        problems.append(f'{len(pkg["files"])} files (max {MAX_FILES}); generate bulk material from setup instead')
    size = sum(len(b.encode()) for b in pkg['files'].values())
    if size > MAX_BYTES:
        problems.append(f'files total {size} bytes (max {MAX_BYTES}); generate bulk material from setup instead')
    words = len(pkg['instruction'].split())
    if not 50 <= words <= 600:
        problems.append(f'instruction has {words} words (allowed 50-600)')
    if problems:
        raise PackageError('; '.join(problems))


ASSET_TAR = 'asset.tar.zst'
SETUP_RUN = 'bash /tasksynth-meta/setup.sh && rm -rf /tasksynth /tasksynth-meta/setup.sh'
# Docker and harbor_ns give each container a fresh /etc/hosts, so a setup that writes it has its copy kept for the
# entrypoint to put back (copied in the same RUN: under Docker, /etc/hosts is a per-RUN mount that never reaches the
# image).
HOSTS = '/etc/tasksynth/hosts'
HOSTS_SETUP_RUN = ('bash /tasksynth-meta/setup.sh && mkdir -p /etc/tasksynth '
                   f'&& {{ [ ! -e /etc/hosts ] || cp /etc/hosts {HOSTS}; }} && rm -rf /tasksynth /tasksynth-meta/setup.sh')
ENTRYPOINT_RUN = ('RUN mv /tasksynth-meta/entrypoint /usr/local/bin/tasksynth-entrypoint '
                  '&& chmod +x /usr/local/bin/tasksynth-entrypoint')
ENTRYPOINT_LINE = 'ENTRYPOINT ["/usr/local/bin/tasksynth-entrypoint"]'


def writes_hosts(pkg):
    return 'etc/hosts' in pkg['setup']


def needs_entrypoint(pkg):
    return bool(pkg['services']) or writes_hosts(pkg)


def dockerfile(pkg, asset=False):
    """asset: the task ships a real asset as environment/tasksynth/asset.tar.zst; it is unpacked to /tasksynth/asset/
    for the setup script and deleted with /tasksynth."""
    meta = pkg['meta']
    lines = [f'FROM {BASE_REF}']
    if meta.get('apt'):
        lines.append('RUN apt-get update && apt-get install -y --no-install-recommends '
                     + ' '.join(meta['apt']) + ' && rm -rf /var/lib/apt/lists/*')
    if meta.get('pip'):
        lines.append('RUN python3 -m pip install --no-cache-dir --break-system-packages '
                     + ' '.join(f"'{p}'" for p in meta['pip']))
    if pkg['files']:
        lines.append('COPY files/ /')
    lines.append('COPY tasksynth/ /tasksynth-meta/')
    setup = HOSTS_SETUP_RUN if writes_hosts(pkg) else SETUP_RUN
    if asset:
        # the tar stays until the final `rm -rf /tasksynth-meta`: harbor_ns retries a failed RUN once, and a retry
        # without it failed on `tar: Cannot open`, hiding the setup's real error from the generator (ra1)
        setup = f'mkdir -p /tasksynth/asset && tar --zstd -xf /tasksynth-meta/{ASSET_TAR} -C /tasksynth/asset && ' + setup
    lines.append('RUN ' + setup)
    if pkg['services']:
        lines.append('RUN mkdir -p /etc/tasksynth && mv /tasksynth-meta/services.sh /etc/tasksynth/services.sh '
                     '&& mv /tasksynth-meta/entrypoint /usr/local/bin/tasksynth-entrypoint '
                     '&& chmod +x /usr/local/bin/tasksynth-entrypoint /etc/tasksynth/services.sh')
    elif writes_hosts(pkg):
        lines.append(ENTRYPOINT_RUN)
    if needs_entrypoint(pkg):
        lines.append(ENTRYPOINT_LINE)
    lines.append('RUN rm -rf /tasksynth-meta')
    lines.append('WORKDIR /app')
    return '\n'.join(lines) + '\n'


ENTRYPOINT = """#!/bin/bash
# Put back the /etc/hosts the task's setup left (the container runtime replaced it, keep its line for this host), start
# the task's daemons, then run the container's command (Harbor: sleep infinity).
if [ -f /etc/tasksynth/hosts ]; then
  h=$(hostname)
  names() { awk -v h="$h" '/^[[:space:]]*#/ {next} {for (i = 2; i <= NF; i++) if ($i == h) {print; next}}' "$1"; }
  own=$(names /etc/hosts)
  cat /etc/tasksynth/hosts > /etc/hosts
  if [ -n "$own" ] && [ -z "$(names /etc/tasksynth/hosts)" ]; then
    [ -z "$(tail -c 1 /etc/hosts)" ] || echo >> /etc/hosts
    printf '%s\\n' "$own" >> /etc/hosts
  fi
fi
if [ -f /etc/tasksynth/services.sh ]; then
  bash /etc/tasksynth/services.sh > /var/log/tasksynth-services.log 2>&1
  rc=$?
  [ "$rc" -eq 0 ] || echo "services failed: exit $rc" >> /var/log/tasksynth-services.log
fi
exec "$@"
"""


FINAL_STATE = Path(__file__).with_name('final_state.py')
TEST_SH = """#!/bin/bash
# tasksynth tasks ship no verifier: record the final state for the trace judge; the reward is always 0.
mkdir -p /logs/verifier
python3 /tests/final_state.py /tests/paths.json > /logs/verifier/final_state.json 2> /logs/verifier/final_state.err
echo 0 > /logs/verifier/reward.txt
"""


def toml_str(s):
    return json.dumps(str(s))


def task_toml(pkg, spec, provenance):
    meta = pkg['meta']
    minutes = int(meta.get('expert_minutes') or spec.get('minutes') or 60)
    tags = [spec.get('domain', ''), spec.get('agent_action', ''), meta.get('kind', '')]
    return f'''schema_version = "1.0"

[metadata]
author_name = "tasksynth"
category = {toml_str(spec.get('agent_action', ''))}
tags = [{', '.join(toml_str(t) for t in tags if t)}]
title = {toml_str(meta.get('title', ''))}
expert_time_estimate_min = {minutes}
tasksynth_spec = {toml_str(spec.get('spec_id', ''))}
tasksynth_generator = {toml_str(provenance.get('model', ''))}

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = {float(max(1800, min(4 * 3600, minutes * 60 * 4)))}

[environment]
network_mode = "no-network"
build_timeout_sec = 1800
cpus = 2
memory_mb = 8192
storage_mb = 10240
gpus = 0
'''


def materialize(pkg, spec, provenance, task_dir, asset_tar=None):
    """Write the package as a Harbor task directory (replacing any previous draft). asset_tar: the real asset's
    tar.zst (task_synthesis.assets.store.task_tar), shipped in the environment for the setup script."""
    task_dir = Path(task_dir)
    if task_dir.exists():
        shutil.rmtree(task_dir)
    env = task_dir / 'environment'
    (env / 'tasksynth').mkdir(parents=True)
    for path, body in pkg['files'].items():
        target = env / 'files' / path.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
        if body.startswith('#!'):
            target.chmod(0o755)
    (env / 'tasksynth' / 'setup.sh').write_text(pkg['setup'] if pkg['setup'].endswith('\n') else pkg['setup'] + '\n')
    if pkg['services']:
        (env / 'tasksynth' / 'services.sh').write_text(pkg['services'] + '\n')
    if needs_entrypoint(pkg):
        (env / 'tasksynth' / 'entrypoint').write_text(ENTRYPOINT)
    if asset_tar:
        shutil.copyfile(asset_tar, env / 'tasksynth' / ASSET_TAR)
    (env / 'Dockerfile').write_text(dockerfile(pkg, asset=bool(asset_tar)))
    (task_dir / 'instruction.md').write_text(pkg['instruction'])
    (task_dir / 'task.toml').write_text(task_toml(pkg, spec, provenance))
    (task_dir / 'solution').mkdir()
    (task_dir / 'solution' / 'sketch.md').write_text(pkg['sketch'])
    (task_dir / 'tests').mkdir()
    (task_dir / 'tests' / 'test.sh').write_text(TEST_SH)
    shutil.copy(FINAL_STATE, task_dir / 'tests' / 'final_state.py')
    (task_dir / 'tests' / 'paths.json').write_text(json.dumps(
        {'inputs': pkg['meta'].get('inputs') or [], 'outputs': pkg['meta'].get('outputs') or []}, indent=1))
    info = {'spec': spec, 'meta': pkg['meta'], 'provenance': provenance}
    if spec.get('asset'):
        info['asset'] = spec['asset']   # source, URL, version, SHA-256, license, attribution (BLUEPRINT: record per asset)
    (task_dir / 'tasksynth.json').write_text(json.dumps(info, indent=1))
    return task_dir


def rehost(task_dir):
    """Give a task materialized before HOSTS existed the current setup step and entrypoint, so the /etc/hosts its setup
    writes survives the container start. Returns True if the task changed. Edits the Dockerfile in place (a setup line
    with a prefix, as in the real-asset rounds, keeps it); one this template did not write raises PackageError."""
    env = Path(task_dir) / 'environment'
    if not ((env / 'Dockerfile').is_file() and (env / 'tasksynth' / 'setup.sh').is_file()):
        raise PackageError(f'{task_dir}: no Dockerfile or setup.sh (a task still being written?)')
    text = (env / 'Dockerfile').read_text()
    if 'etc/hosts' not in (env / 'tasksynth' / 'setup.sh').read_text() or HOSTS_SETUP_RUN in text:
        return False
    lines = text.splitlines()
    at = [i for i, line in enumerate(lines) if line.startswith('RUN ') and line.endswith(SETUP_RUN)]
    if len(at) != 1 or any(line.startswith('ENTRYPOINT') and line != ENTRYPOINT_LINE for line in lines):
        raise PackageError(f'{task_dir}: the Dockerfile is not this template\'s')
    lines[at[0]] = lines[at[0]][:-len(SETUP_RUN)] + HOSTS_SETUP_RUN
    if ENTRYPOINT_LINE not in lines:
        lines[at[0] + 1:at[0] + 1] = [ENTRYPOINT_RUN, ENTRYPOINT_LINE]
    (env / 'tasksynth' / 'entrypoint').write_text(ENTRYPOINT)
    (env / 'Dockerfile').write_text('\n'.join(lines) + '\n')
    return True
