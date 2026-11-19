"""Minimal OCI registry client: fetch an image's config (WorkingDir, Env, Entrypoint, Cmd, User).

Apptainer's docker:// conversion keeps Env (in 10-docker2singularity.sh) but drops
WorkingDir and makes Entrypoint/Cmd hard to recover, so harbor_ns reads the
config blob itself.  Anonymous bearer tokens, plus Docker Hub credentials from
~/.docker/config.json for Docker Hub.  Uses the environment's HTTP(S) proxy.
"""
from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request

ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def parse_ref(ref: str) -> tuple[str, str, str]:
    """-> (registry host, repository, reference)"""
    name, digest = (ref.split("@", 1) + [None])[:2]
    first = name.split("/", 1)[0]
    if "/" in name and ("." in first or ":" in first or first == "localhost"):
        host, repo = name.split("/", 1)
    else:
        host, repo = "registry-1.docker.io", name
    if host in ("docker.io", "index.docker.io"):
        host = "registry-1.docker.io"
    tag = "latest"
    last = repo.rsplit("/", 1)[-1]
    if ":" in last:
        repo, tag = repo.rsplit(":", 1)
    if host == "registry-1.docker.io" and "/" not in repo:
        repo = "library/" + repo
    return host, repo, digest or tag


def _hub_basic() -> str | None:
    try:
        d = json.load(open(os.path.expanduser("~/.docker/config.json")))
        return d["auths"]["https://index.docker.io/v1/"]["auth"]
    except Exception:  # noqa: BLE001
        return None


class _Client:
    def __init__(self, host: str, repo: str):
        self.host, self.repo, self.token = host, repo, None

    def _auth(self, header: str) -> None:
        params = dict(re.findall(r'(\w+)="([^"]*)"', header))
        realm = params.pop("realm")
        params.setdefault("scope", f"repository:{self.repo}:pull")
        url = realm + "?" + "&".join(f"{k}={urllib.request.quote(v, safe=':/')}" for k, v in params.items())
        req = urllib.request.Request(url)
        basic = _hub_basic() if self.host == "registry-1.docker.io" else None
        if basic:
            req.add_header("Authorization", "Basic " + basic)
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        self.token = d.get("token") or d.get("access_token")

    def get(self, path: str, accept: str | None = None) -> tuple[bytes, str]:
        url = f"https://{self.host}/v2/{self.repo}/{path}"
        for _ in range(2):
            req = urllib.request.Request(url)
            if accept:
                req.add_header("Accept", accept)
            if self.token:
                req.add_header("Authorization", "Bearer " + self.token)
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read(), r.headers.get("Content-Type", "")
            except urllib.error.HTTPError as e:
                if e.code == 401 and "WWW-Authenticate" in e.headers and not self.token:
                    self._auth(e.headers["WWW-Authenticate"])
                    continue
                raise
        raise RuntimeError(f"unauthorized: {url}")


def image_config(ref: str) -> dict:
    """Return the image config's `config` section (WorkingDir, Env, Entrypoint, Cmd, User, ...)."""
    host, repo, reference = parse_ref(ref)
    c = _Client(host, repo)
    body, ctype = c.get(f"manifests/{reference}", ACCEPT)
    m = json.loads(body)
    if "manifests" in m:  # index / manifest list: pick linux/amd64
        cands = [x for x in m["manifests"] if (x.get("platform") or {}).get("architecture") == "amd64"
                 and (x.get("platform") or {}).get("os") == "linux"]
        if not cands:
            raise RuntimeError(f"no linux/amd64 manifest for {ref}")
        body, ctype = c.get(f"manifests/{cands[0]['digest']}", ACCEPT)
        m = json.loads(body)
    cfg_digest = m["config"]["digest"]
    body, _ = c.get(f"blobs/{cfg_digest}")
    return json.loads(body).get("config") or {}
