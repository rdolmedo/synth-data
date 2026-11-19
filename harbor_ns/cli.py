"""harbor_ns command line: build task images and poke at sandboxes.

  python -m harbor_ns.cli build <env_dir> [--label X]
  python -m harbor_ns.cli shell <env_dir> -- <command>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from .builder import ImageStore
from .sandbox import RuntimeConfig, Sandbox, build_nsfake, find_runtime_bins, proxy_env


def default_store() -> ImageStore:
    local = Path(os.environ.get("HARBOR_NS_LOCAL", f"/tmp/{os.environ.get('USER', 'user')}/harbor_ns"))
    shared = os.environ.get("HARBOR_NS_SHARED", "/data/aai/tbench/harbor_ns")
    bins = local / "bin"
    rt = RuntimeConfig(nsinit=find_runtime_bins(bins), work_root=local / "boxes", nsfake=build_nsfake(bins))
    rt.work_root.mkdir(parents=True, exist_ok=True)
    return ImageStore(local_root=local, shared_root=Path(shared) if shared else None, rt=rt,
                      nsfake=rt.nsfake)


async def _build(a):
    st = default_store()
    t0 = time.time()
    key, rootfs, meta = await st.build(Path(a.env_dir), label=a.label)
    print(json.dumps({"key": key, "rootfs": str(rootfs), "seconds": round(time.time() - t0, 1),
                      "workdir": meta.workdir, "entrypoint": meta.entrypoint, "user": meta.user}))


async def _shell(a):
    st = default_store()
    key, rootfs, meta = await st.build(Path(a.env_dir), label=a.label)
    box = Sandbox(cfg=st.rt, lower=[rootfs], hostname="test",
                  base_env={**meta.env, **proxy_env()})
    await box.start()
    try:
        r = await box.exec(" ".join(a.cmd), cwd=meta.workdir, timeout=600)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        print(f"[rc={r.return_code} proc={box.proc_mode}]")
    finally:
        await box.stop()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="op", required=True)
    b = sub.add_parser("build")
    b.add_argument("env_dir")
    b.add_argument("--label", default="")
    s = sub.add_parser("shell")
    s.add_argument("env_dir")
    s.add_argument("--label", default="")
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.op == "shell" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    asyncio.run(_build(a) if a.op == "build" else _shell(a))


if __name__ == "__main__":
    main()
