"""Finite, checkpointed serving sweep. Never reuses or terminates another job's server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from replay_bench import scrape, server_summary

HERE = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def probe(base):
    common = {"model": "nemotron-3.5-super-vl", "temperature": 0, "max_tokens": 2048}
    text = request(base + "/chat/completions", {**common, "messages": [
        {"role": "user", "content": "What is 17 + 25? Answer briefly."}]}, timeout=600)
    msg = text["choices"][0]["message"]
    if "42" not in (msg.get("content") or ""):
        raise RuntimeError(f"Text probe failed: {text}")
    tool = request(base + "/chat/completions", {**common, "messages": [
        {"role": "user", "content": "Use bash to run exactly: printf NEMOTRON_TOOL_OK"}],
        "tools": [{"type": "function", "function": {"name": "bash", "description": "Run shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]}, timeout=600)
    calls = tool["choices"][0]["message"].get("tool_calls") or []
    if not calls or calls[0]["function"]["name"] != "bash":
        raise RuntimeError(f"Tool parsing probe failed: {tool}")
    json.loads(calls[0]["function"]["arguments"])
    editor_schema = json.loads((HERE.parent / "eval/configs/file_editor_tool.json").read_text())
    editor = request(base + "/chat/completions", {**common, "messages": [
        {"role": "user", "content": "Call file_editor with command view and path /tmp/example.txt."}],
        "tools": [editor_schema]}, timeout=600)
    ecalls = editor["choices"][0]["message"].get("tool_calls") or []
    if not ecalls or ecalls[0]["function"]["name"] != "file_editor":
        raise RuntimeError(f"Editor tool parsing probe failed: {editor}")
    arguments = json.loads(ecalls[0]["function"]["arguments"])
    if arguments.get("command") != "view" or arguments.get("path") != "/tmp/example.txt":
        raise RuntimeError(f"Editor arguments probe failed: {arguments}")
    return {"text": text, "tool": tool, "file_editor": editor}


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", choices=["hgpu100", "b200"], required=True)
    ap.add_argument("--mode", choices=["study", "serve", "smoke", "admission"], default="study")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--levels", default=os.environ.get("BENCH_LEVELS", "64:300:480,128:300:480,256:420:600,512:420:600"))
    ap.add_argument("--max-clients", type=int, default=2048)
    ap.add_argument("--mtps", default=os.environ.get("BENCH_MTPS", ""))
    a = ap.parse_args()
    mtps = [3, 1, 0] if a.hardware == "hgpu100" else [5, 1, 0]
    if a.mtps:
        mtps = [int(n) for n in a.mtps.split(",")]
    if a.mode != "study":
        mtps = [int(os.environ.get("MTP_TOKENS", mtps[0]))]
    for mtp in mtps:
        run = a.out / f"mtp{mtp}"
        run.mkdir(parents=True, exist_ok=True)
        if (run / "COMPLETE").exists() and a.mode == "study":
            continue
        plan_path = run / "replay-plan.json"
        if a.mode == "study" and not plan_path.exists():
            plan_path.write_text(json.dumps({"levels": a.levels,
                "extend_max_clients": a.max_clients}, indent=2) + "\n")
        port = free_port()
        base = f"http://127.0.0.1:{port}/v1"
        env = {**os.environ, "MTP_TOKENS": str(mtp), "HARDWARE": a.hardware}
        cmd = ["bash", str(HERE / "serve_nemotron.sh"), str(port)]
        if a.mode == "admission":
            # This isolated study owns the server; expose cache reset only on
            # localhost and never change the live evaluation server.
            env["VLLM_SERVER_DEV_MODE"] = "1"
            cmd.extend(["--host", "127.0.0.1"])
        launcher = (HERE / "serve_nemotron.sh").read_bytes()
        (run / "serve_nemotron.sh").write_bytes(launcher)
        (run / "launch.json").write_text(json.dumps({"command": cmd, "mtp": mtp,
            "hardware": a.hardware, "model_revision": "82d93699acd3be9e4617c53360fe5b3f5dbeb1b3",
            "vllm_commit": "3f1bcfd82bb2e93fcd364fc7a5491718453a77e2",
            "worktree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
            "launcher_sha256": hashlib.sha256(launcher).hexdigest(),
            "omp_num_threads": env.get("OMP_NUM_THREADS"),
            "serving_seed": int(env.get("SERVING_SEED", "0")),
            "levels": a.levels, "workload": "GLM seed-trajectory replay; NOT live evaluation",
            "replay_plan": json.loads(plan_path.read_text()) if plan_path.exists() else None,
            "started": time.time()}, indent=2))
        with (run / "vllm.log").open("a") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 3600
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"vLLM exited with {proc.returncode}; see {run / 'vllm.log'}")
                    try:
                        request(base + "/models")
                        break
                    except (OSError, ValueError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("vLLM startup exceeded one hour")
                        time.sleep(5)
                (run / "probe.json").write_text(json.dumps(probe(base), indent=2))
                print(f"Ready: {a.hardware} MTP{mtp} on {socket.gethostname()}:{port}", flush=True)
                if a.mode == "admission":
                    with (run / "bench.log").open("a") as benchlog:
                        subprocess.run([sys.executable, str(HERE / "nemotron_admission_replay.py"),
                            "--base", base, "--out", str(run)],
                            check=True, stdout=benchlog, stderr=subprocess.STDOUT)
                    (run / "COMPLETE").write_text(f"{time.time()}\n")
                elif a.mode == "serve":
                    endpoint = a.out / "endpoint.url"
                    host = socket.gethostname()
                    if "." not in host:
                        host += ".internal.cluster.internal"
                    endpoint.write_text(f"http://{host}:{port}/v1\n")
                    try:
                        previous, last = scrape(base.removesuffix("/v1")), time.monotonic()
                        while proc.poll() is None:
                            time.sleep(30)
                            current, now = scrape(base.removesuffix("/v1")), time.monotonic()
                            if current and previous:
                                sample = {"timestamp": time.time(), "window_sec": now - last,
                                    "raw": current, **server_summary([previous], [current], now - last)}
                                with (run / "live_metrics.jsonl").open("a") as f:
                                    f.write(json.dumps(sample) + "\n")
                            previous, last = current, now
                        raise RuntimeError(f"Serving process exited: {proc.returncode}")
                    finally:
                        endpoint.unlink(missing_ok=True)
                else:
                    levels = "2:60:120" if a.mode == "smoke" else a.levels
                    with (run / "bench.log").open("a") as benchlog:
                        subprocess.run([sys.executable, str(HERE / "replay_bench.py"),
                            "--model", "nemotron-3.5-super-vl", "--bases", base,
                            "--metrics-bases", base.removesuffix("/v1"), "--levels", levels,
                            "--results", "/data/aai/tbench/runs/glm-v1/results/*,/data/aai/tbench/runs/glm-v2/results/*,/data/aai/tbench/runs/glm-v3/results/*,/data/aai/tbench/runs/glm-v5/results/*",
                            "--random-start", "--tool-dist", "lognormal", "--cap-tokens", "132000",
                            "--seed", "0", "--out", str(run / "results.jsonl"),
                            "--tag", f"{a.out.name}-mtp{mtp}"],
                            check=True, stdout=benchlog, stderr=subprocess.STDOUT)
                    (run / "COMPLETE").write_text(f"{time.time()}\n")
            finally:
                stop(proc)


if __name__ == "__main__":
    main()
