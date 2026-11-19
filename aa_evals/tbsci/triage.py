#!/usr/bin/env python3
"""Static triage of the TB-Science tasks: resources, network, verifier and harbor_ns compatibility.

    python -I triage.py --tasks <unpacked tasks.tar>/tasks \\
        --hub-digests /data/aai/aa_evals/data/tbsci/v0.1.0/hub_v0.1.json \\
        --manifest /data/aai/aa_evals/data/tbsci/v0.1.0/manifest.json \\
        --out /data/aai/aa_evals/data/tbsci/v0.1.0/triage.json

Run it with the Harbor venv's python (it parses task.toml with Harbor's own TaskConfig and hashes
each task with Harbor's Packager, which is how the Hub digests are computed).  `--tasks` is the
flattened layout `<tasks>/<slug>/task.toml` that pack.py writes.

Flags per task (`issues`):
  gpu                  gpus > 0: harbor_ns has no GPU support
  agent_allowlist      [agent]/[environment] network_mode = "allowlist": harbor_ns rejects it
  compose_main_image   a compose service other than `main` uses ${MAIN_IMAGE_NAME}: harbor_ns treats it as
                       a registry image (it is the task's own build, which only Docker's compose resolves)
  verifier_allowlist   the separate verifier asks for an allowlist
  sidecar_artifacts    artifacts or verifier collect hooks read from a compose sidecar: plain harbor_ns lacks
                       Harbor's service_* operations, so the verifier silently gets nothing (use
                       aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment)
  digest_mismatch      Harbor content hash differs from the Hub v0.1 digest
"""
import argparse
import json
import re
from pathlib import Path

from harbor.models.task.config import TaskConfig
from harbor.publisher.packager import Packager

MAIN_IMAGE_RE = re.compile(r"\$\{?MAIN_IMAGE_NAME\}?")


def compose_info(d: Path) -> dict | None:
    import yaml

    for name in ("docker-compose.yaml", "docker-compose.yml"):
        f = d / name
        if f.exists():
            data = yaml.safe_load(f.read_text()) or {}
            svcs = data.get("services") or {}
            main_img_users = [s for s, v in svcs.items()
                              if s != "main" and MAIN_IMAGE_RE.search(json.dumps(v or {}))]
            return {"file": name, "services": sorted(svcs), "main_image_reused_by": main_img_users}
    return None


def mode(x):
    return getattr(x, "value", x)


def triage(task_dir: Path, hub: dict, src_path: str | None = None, patched_digest: str | None = None) -> dict:
    cfg = TaskConfig.model_validate_toml((task_dir / "task.toml").read_text())
    raw = cfg.model_dump(mode="json")
    env, agent, ver = cfg.environment, cfg.agent, cfg.verifier
    md = raw.get("metadata") or {}
    env_dir, tests_dir = task_dir / "environment", task_dir / "tests"
    comp = compose_info(env_dir)
    vcomp = compose_info(tests_dir)
    venv = ver.environment
    rec = {
        "task": task_dir.name,
        # the repo path tasks/<domain>/<field>/<slug> is canonical; task.toml spellings vary
        "domain": src_path.split("/")[1] if src_path else md.get("domain"),
        "field": src_path.split("/")[2] if src_path else md.get("field"), "subfield": md.get("subfield"),
        "expert_hours": md.get("expert_time_estimate_hours"),
        "agent_timeout_min": (agent.timeout_sec or 0) / 60,
        "verifier_timeout_min": (ver.timeout_sec or 0) / 60,
        "build_timeout_min": (env.build_timeout_sec or 0) / 60,
        "cpus": env.cpus, "memory_gb": (env.memory_mb or 0) / 1024, "storage_gb": (env.storage_mb or 0) / 1024,
        "gpus": env.gpus or 0, "gpu_types": env.gpu_types,
        "docker_image": env.docker_image,
        "env_network": mode(env.network_mode),
        "agent_network": mode(agent.network_mode) if agent.network_mode is not None else None,
        "agent_allowed_hosts": agent.allowed_hosts or env.allowed_hosts,
        "verifier_mode": mode(ver.environment_mode),
        "verifier_network": mode(ver.network_mode) if ver.network_mode is not None else (
            mode(venv.network_mode) if venv is not None else None),
        "verifier_cpus": venv.cpus if venv is not None else None,
        "verifier_memory_gb": (venv.memory_mb or 0) / 1024 if venv is not None else None,
        "verifier_gpus": (venv.gpus or 0) if venv is not None else 0,
        "env_dockerfile": (env_dir / "Dockerfile").exists(),
        "env_compose": comp,
        "tests_dockerfile": (tests_dir / "Dockerfile").exists(),
        "tests_compose": vcomp,
        "artifacts": raw.get("artifacts"),
        "mcp_servers": len(env.mcp_servers or []),
        "multi_step": bool(raw.get("steps")),
    }
    digest, files = Packager.compute_content_hash(task_dir)
    rec["content_hash"] = f"sha256:{digest}"
    rec["n_files"] = len(files)
    rec["bytes"] = sum(f.stat().st_size for f in files)
    issues = []
    if rec["gpus"] or rec["verifier_gpus"]:
        issues.append("gpu")
    if "allowlist" in (rec["env_network"], rec["agent_network"]):
        issues.append("agent_allowlist")
    if rec["verifier_network"] == "allowlist":
        issues.append("verifier_allowlist")
    if (comp and comp["main_image_reused_by"]) or (vcomp and vcomp["main_image_reused_by"]):
        issues.append("compose_main_image")
    sidecar = sorted({a.get("service") for a in (raw.get("artifacts") or [])
                      if isinstance(a, dict) and a.get("service") not in (None, "main")}
                     | {c.get("service") for c in ((raw.get("verifier") or {}).get("collect") or [])
                        if c.get("service") not in (None, "main")})
    rec["sidecar_services"] = sidecar
    if sidecar:
        issues.append("sidecar_artifacts")
    want = patched_digest or hub.get(task_dir.name)  # patched tasks: the digest fetch.py recorded
    if want is None or want != rec["content_hash"]:
        issues.append("digest_mismatch")
    rec["hub_digest"] = want
    rec["issues"] = issues
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, type=Path)
    ap.add_argument("--hub-digests", required=True, type=Path, help="json {slug: 'sha256:...'}")
    ap.add_argument("--manifest", type=Path, help="fetch.py's manifest.json (source paths -> domain/field)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    hub = json.loads(a.hub_digests.read_text())
    man = json.loads(a.manifest.read_text())["tasks"] if a.manifest else {}
    recs = [triage(d, hub, man.get(d.name, {}).get("path"), man.get(d.name, {}).get("patched_digest"))
            for d in sorted(p.parent for p in a.tasks.glob("*/task.toml"))]
    a.out.write_text(json.dumps(recs, indent=1))
    n = len(recs)
    print(f"{n} tasks")
    for flag in ("gpu", "agent_allowlist", "verifier_allowlist", "compose_main_image", "sidecar_artifacts",
                 "digest_mismatch"):
        hit = [r["task"] for r in recs if flag in r["issues"]]
        print(f"  {flag}: {len(hit)} {hit}")


if __name__ == "__main__":
    main()
