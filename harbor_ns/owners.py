"""Build-time file owners for multi-uid sandboxes (port of the tbsci-lite branch's subids owner replay).

Image builds stay single-uid: a Dockerfile `chown agent /app` (or `COPY --chown`, `USER x` + `RUN mkdir`) runs under
nsfake, which records the owner in the xattr ``user.harborns.owner`` and leaves the file root-owned.  A single-uid
sandbox (nsfake) reports the recorded owner; a multi-uid sandbox has real ids and no nsfake, so without a replay
every such file is root-owned there: the agent user can't write its own /app (unknown-payload-balancing), a
verifier user can't read a `root:65532 0550` /tests (pcasl-cbf-quantifier).  This module applies the recorded
owners for real, inside the pod's user namespace, on the sandbox's overlay (the cached image is untouched).

Where the records come from: a locally built rootfs keeps the xattrs (scanned, cached per layer list); an image
unpacked from the shared squashfs cache (unsquashfs -no-xattrs: a rootless unpack can't set them all) gets
``owners.json`` next to its rootfs, read from a read-only squashfuse mount of the squashfs at unpack time.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

XATTR = "user.harborns.owner"
HELPER = Path(__file__).resolve()
SKIP_TOP = {"proc", "sys", "dev"}


def wanted() -> bool:
    """Owner records are only needed by multi-uid sandboxes (HARBOR_NS_MULTIUID=1 in the process that unpacks)."""
    return os.environ.get("HARBOR_NS_MULTIUID", "").strip().lower() in ("1", "true", "yes")


def _scan(top: Path) -> dict[str, tuple[int, int] | None]:
    seen: dict[str, tuple[int, int] | None] = {}
    for root, dirs, files in os.walk(top):
        rel_root = os.path.relpath(root, top)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d not in SKIP_TOP]
        for name in [*dirs, *files]:
            rel = "/" + os.path.normpath(os.path.join(rel_root, name)).lstrip("./")
            try:
                v = os.getxattr(os.path.join(root, name), XATTR, follow_symlinks=False).decode()
                u, g = (int(x) for x in v.split(":"))
                seen[rel] = (u, g)
            except (OSError, ValueError):
                seen[rel] = None
    return seen


def owners_from_sqfs(sqfs: Path, out: Path, log: Path | None = None) -> None:
    """Write owners.json for an image unpacked without xattrs, from a squashfuse mount of its squashfs."""
    import tempfile  # noqa: PLC0415

    def note(msg: str) -> None:
        if log is not None:
            with open(log, "a") as f:
                f.write(f"[owners] {msg}\n")

    mnt = Path(tempfile.mkdtemp(prefix="owners-mnt-", dir=out.parent))
    try:
        r = subprocess.run(["squashfuse", str(sqfs), str(mnt)], capture_output=True, text=True)
        if r.returncode != 0:
            note(f"squashfuse failed, no owners.json: {r.stderr.strip()[:300]}")
            return
        try:
            owners = sorted((p, ug[0], ug[1]) for p, ug in _scan(mnt).items() if ug)
        finally:
            subprocess.run(["fusermount", "-u", str(mnt)], capture_output=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(owners))
        os.replace(tmp, out)
        note(f"owners.json: {len(owners)} recorded owners")
    finally:
        try:
            mnt.rmdir()
        except OSError:
            pass


def recorded(lower: list[Path], cache_dir: Path) -> list[tuple[str, int, int]] | None:
    """(path in the image, uid, gid) of every file nsfake gave an owner at build time; None when an unpacked
    image has no owners.json (unknown, not "none")."""
    for layer in lower:
        oj = Path(layer).parent / "owners.json"
        if Path(layer).name == "rootfs" and oj.is_file():
            try:
                return [tuple(x) for x in json.loads(oj.read_text())]
            except ValueError:
                return None
    key = hashlib.sha256("\n".join(str(p) for p in lower).encode()).hexdigest()[:24]
    cache = cache_dir / f"owners-{key}.json"
    try:
        return [tuple(x) for x in json.loads(cache.read_text())]
    except (OSError, ValueError):
        pass
    seen: dict[str, tuple[int, int] | None] = {}
    for layer in lower:  # top first: the first layer that has a path decides
        for rel, ug in _scan(Path(layer)).items():
            seen.setdefault(rel, ug)
    out = sorted((p, ug[0], ug[1]) for p, ug in seen.items() if ug)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out))
        os.replace(tmp, cache)
    except OSError:
        pass
    return out


def replay(holder_pid: int, root: str, owners: list[tuple[str, int, int]]) -> str:
    """chown the recorded owners for real, as the pod's namespace root (host python, this file's CLI)."""
    if not owners:
        return ""
    r = subprocess.run(["nsenter", "-t", str(holder_pid), "-U", "--preserve-credentials", "--",
                        sys.executable, "-I", str(HELPER), "chown-list", root],
                       input=json.dumps(owners), capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def _chown_keep_mode(path: str, u: int, g: int) -> None:
    st = os.lstat(path)
    os.lchown(path, u, g)
    # chown clears set-id bits on regular files, even for root; Docker keeps them.
    if not os.path.islink(path) and st.st_mode & 0o6000:
        os.chmod(path, st.st_mode & 0o7777)


def _main(argv: list[str]) -> int:
    if argv[:1] != ["chown-list"]:
        return 2
    root, errors, done = argv[1].rstrip("/"), 0, 0
    for p, u, g in json.loads(sys.stdin.read() or "[]"):
        try:
            _chown_keep_mode(root + p, u, g)
            done += 1
        except OSError:
            errors += 1
    print(f"owners replayed: {done}, errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
