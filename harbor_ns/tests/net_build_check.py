"""Build a TMax-shaped image through a proxy that refuses connections for a while (issue #7).

  HARBOR_NS_LOCAL=/node/local/dir python harbor_ns/tests/net_build_check.py --down 60 [--apt-retries N]
      [--backoff S] [--out result.json]

Runs whichever harbor_ns is first on sys.path / PYTHONPATH, so the same scenario compares
branches. A local TCP relay stands in for the cluster proxy: it resets every connection during
the first --down seconds (the 2026-10-23 symptom: "Unable to connect to 10.22.0.107:8080",
"Connection reset by peer") and afterwards forwards to the real proxy. The task mirrors TMax:
post_install.sh runs `apt-get update && apt-get install` without `set -e`, and the next RUN needs
the package. Nothing is written to the shared harbor_ns store: shared_root is a scratch dir seeded
with the ubuntu:22.04 base squashfs.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import sys
import tempfile
import time

REAL_PROXY = ('proxy', 8080)
BASE = Path('/data/aai/tbench/harbor_ns/bases/ubuntu_22.04')  # what TMax `FROM ubuntu:22.04` resolves to

DOCKERFILE = """FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

COPY post_install.sh /tmp/post_install.sh
RUN bash /tmp/post_install.sh && rm /tmp/post_install.sh
RUN jq --version
"""
# No `set -e`, last command succeeds: an apt failure is invisible to this step (as in TMax).
POST_INSTALL = """apt-get update && apt-get install -y --no-install-recommends jq
mkdir -p /home/user
echo setup done
"""


class FlakyRelay:
    def __init__(self, down_seconds, upstream):
        self.down_until = time.time() + down_seconds
        self.upstream = upstream
        self.reset = self.forwarded = 0

    async def handle(self, reader, writer):
        if time.time() < self.down_until:
            self.reset += 1
            sock = writer.get_extra_info('socket')
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))  # RST on close
            writer.close()
            return
        self.forwarded += 1
        try:
            ur, uw = await asyncio.open_connection(*self.upstream)
        except OSError:
            writer.close()
            return

        async def pump(r, w):
            try:
                while data := await r.read(65536):
                    w.write(data)
                    await w.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                try:
                    w.close()
                except OSError:
                    pass

        await asyncio.gather(pump(reader, uw), pump(ur, writer))


async def main(a):
    relay = FlakyRelay(a.down, (socket.gethostbyname(REAL_PROXY[0]), REAL_PROXY[1]))
    server = await asyncio.start_server(relay.handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    os.environ['HARBOR_NS_PROXY_HOST'] = '127.0.0.1'
    os.environ['HARBOR_NS_PROXY_PORT'] = str(port)
    from harbor_ns import builder
    from harbor_ns.sandbox import RuntimeConfig, build_nsfake, find_runtime_bins
    if a.apt_retries is not None and hasattr(builder, 'APT_BUILD_CONF'):
        builder.APT_BUILD_CONF = builder.APT_BUILD_CONF.replace('Acquire::Retries "10"', f'Acquire::Retries "{a.apt_retries}"')
    local = Path(os.environ.get('HARBOR_NS_LOCAL') or tempfile.mkdtemp()) / f'netcheck-{os.getpid()}'
    shared = local / 'shared'
    (shared / 'bases').mkdir(parents=True)
    for ext in ('.sqfs', '.json'):
        os.symlink(f"{BASE}{ext}", shared / "bases" / (BASE.name + ext))
    ctx = local / 'ctx'
    ctx.mkdir()
    (ctx / 'Dockerfile').write_text(DOCKERFILE)
    (ctx / 'post_install.sh').write_text(POST_INSTALL)
    bins = local / 'bin'
    rt = RuntimeConfig(nsinit=find_runtime_bins(bins), work_root=local / 'boxes', nsfake=build_nsfake(bins))
    rt.work_root.mkdir(parents=True, exist_ok=True)
    kw = {}
    if a.backoff is not None and 'network_backoff' in builder.ImageStore.__dataclass_fields__:
        kw['network_backoff'] = (a.backoff,)
    store = builder.ImageStore(local_root=local / 'store', shared_root=shared, rt=rt, nsfake=rt.nsfake, **kw)
    t0 = time.time()
    await store.ensure_base('ubuntu:22.04')  # unpack the base before the clock that matters
    t1 = time.time()
    outcome = {'harbor_ns': str(Path(builder.__file__).parent), 'down_seconds': a.down, 'apt_retries': a.apt_retries}
    try:
        key, rootfs, meta = await store.build(ctx, label='netcheck')
        outcome.update(ok=True, jq=(rootfs / 'usr/bin/jq').exists())
        md = json.loads((store.image_dir(key) / 'meta.json').read_text())
        outcome.update(build_attempts=md.get('build_attempts', 1), proxy_failures_last_attempt=md.get('proxy_failures_last_attempt', 0))
        log = store.image_dir(key) / 'build.log'
    except builder.BuildError as e:
        outcome.update(ok=False, error=str(e))
        logs = list((shared / 'logs').glob('*.failed.log'))
        log = logs[0] if logs else None
    outcome.update(build_seconds=round(time.time() - t1, 1), base_seconds=round(t1 - t0, 1),
                   relay_resets=relay.reset, relay_forwarded=relay.forwarded)
    if log and log.exists():
        text = log.read_text(errors='replace')
        outcome['log_tail'] = [l for l in text.splitlines() if l.startswith(('W:', 'E:', 'Err:', '[harbor_ns]'))
                               or 'not found' in l or 'jq-' in l][-12:]
        if hasattr(builder, 'proxy_failures'):
            outcome['proxy_failures_in_log'] = builder.proxy_failures(text, port)
    server.close()
    print(json.dumps(outcome, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(outcome, indent=1))
    if not a.keep:
        import subprocess
        subprocess.run(['chmod', '-R', 'u+rwX', str(local)], check=False)
        shutil.rmtree(local, ignore_errors=True)
    return 0 if outcome['ok'] else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--down', type=float, default=60, help='seconds the relay resets every connection')
    ap.add_argument('--apt-retries', type=int, help='override Acquire::Retries in the build config (new code only)')
    ap.add_argument('--backoff', type=float, help='seconds before a network rebuild (new code only)')
    ap.add_argument('--out')
    ap.add_argument('--keep', action='store_true')
    sys.exit(asyncio.run(main(ap.parse_args())))
