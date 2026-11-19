"""Opt-in private Pi runtime for one musl task; task binaries keep their own libc."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

TASK = 'unknown-payload-balancing'
COMPAT_SHA256 = '137b375210c63344b5d15048b2e2307d2cbd9b6d92121ae5d88fc1677a6c6633'
WRAPPER = ('#!/bin/sh\nexec /opt/harbor-pi/glibc-compat/ld-linux-x86-64.so.2 '
           '--library-path /opt/harbor-pi/glibc-compat '
           '/opt/harbor-pi/node/bin/node '
           '/opt/harbor-pi/pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js "$@"\n')


async def install(agent, environment):
    from tb_agents.harness_agents import _sandbox_platform, _sha256

    if environment.environment_name != TASK:
        raise ValueError('Pi musl compatibility adapter is restricted to ' + TASK)
    platform = await _sandbox_platform(environment)
    if platform != 'x86_64-musl':
        raise ValueError('Pi musl compatibility adapter requires x86_64-musl, got ' + repr(platform))
    archive = Path(os.environ.get('PI_RUNTIME_COMPAT_ARCHIVE') or '')
    if not archive.is_file():
        raise ValueError('Pi musl compatibility archive is required')
    sidecar = archive.with_suffix('.tar.sha256')
    tokens = sidecar.read_text().split() if sidecar.is_file() else []
    if not tokens or tokens[0] != COMPAT_SHA256 or _sha256(archive) != COMPAT_SHA256:
        raise ValueError('Pi musl compatibility archive checksum mismatch')
    # This inherited selector verifies the unchanged patched Pi bundle and local
    # staged copy. Only this guarded installer translates the musl platform.
    source = agent._cached_bundle('x86_64')
    if source is None:
        raise ValueError('Pi musl compatibility adapter requires the pinned patched Pi bundle')
    await agent.ensure_system_dependencies(environment, ('bash', 'tar', 'coreutils'))
    await environment.upload_file(source, '/tmp/harbor-pi.tar')
    await environment.upload_file(archive, '/tmp/harbor-pi-glibc-compat.tar')
    await agent.exec_as_root(environment, command=(
        'tar -xf /tmp/harbor-pi.tar -C / && tar -xf /tmp/harbor-pi-glibc-compat.tar -C / && '
        'rm -f /tmp/harbor-pi.tar /tmp/harbor-pi-glibc-compat.tar && mkdir -p /usr/local/bin && '
        'printf %s ' + shlex.quote(WRAPPER) + ' > /opt/harbor-pi/bin-pi && '
        'chmod 755 /opt/harbor-pi/bin-pi && ln -sf /opt/harbor-pi/bin-pi /usr/local/bin/pi'))
    record = {'adapter': 'PiCont16kEnvClockMuslCompat', 'task': TASK, 'platform': platform,
              'archive': str(archive), 'sha256': COMPAT_SHA256, 'wrapper': WRAPPER,
              'pi_bundle_sha256': agent._continuation_sha256,
              'pinned_pi_bundle_unchanged': True, 'global_library_env_changed': False,
              'task_source_changed': False, 'status': 'installed_version_check_pending'}
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    path = agent.logs_dir / 'pi-runtime-compat.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    version = await agent.exec_as_agent(environment, command='pi --version')
    record.update(status='ready', version_stdout=version.stdout)
    path.write_text(json.dumps(record, indent=2) + '\n')


async def upload_config(agent, environment, upload, *, content, remote_path, filename):
    """Give private Pi configs to the actual image user without changing task defaults."""
    import re
    if environment.environment_name != TASK or Path(remote_path).parent != Path('/tmp/harbor-pi-agent'):
        raise ValueError('Pi musl config ownership is restricted to this task and private config directory')
    previous = environment.default_user
    result = await agent.exec_as_agent(environment, command='id -u')
    value = (result.stdout or '').strip()
    if re.fullmatch(r'[0-9]+', value) is None or not 0 <= int(value) < 2**32 - 1:
        raise ValueError('Cannot resolve numeric effective Pi agent UID')
    uid = int(value)
    path = agent.logs_dir / 'pi-runtime-compat.json'
    record = json.loads(path.read_text())
    if record.get('status') != 'ready' or record.get('task') != TASK or record.get('platform') != 'x86_64-musl':
        raise ValueError('Pi musl config ownership requires a verified installed runtime')
    event = {'remote_path': remote_path, 'effective_uid': uid, 'default_user_before': previous}
    try:
        # Harbor's existing uploader performs chown+0600 only for an explicit
        # default user. Image USER otherwise applies to exec but not uploads.
        with environment.with_default_user(uid):
            await upload(environment, content=content, remote_path=remote_path, filename=filename)
        event['status'] = 'uploaded'
    except BaseException as exc:
        event.update(status='upload_failed', error_type=type(exc).__name__)
        raise
    finally:
        event['default_user_after'] = environment.default_user
        record.setdefault('config_ownership', []).append(event)
        path.write_text(json.dumps(record, indent=2) + '\n')
