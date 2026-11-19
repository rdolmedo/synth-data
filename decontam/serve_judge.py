"""Standalone DeepSeek-V4.1-Flash-1017 judge server (issue #14): the production teacher command, no campaign.

  python -m decontam.serve_judge submit --home DIR      (login node: writes DIR/serve.sub, condor_submit_bid 63)
  python -m decontam.serve_judge run --home DIR         (inside the 4xB200 job)

The engine is synth/teachers/deepseek-v41-flash-1017.toml's command and env with issue #9's fixes:
rejection_sample_method "standard" (vllm#61114: "block" biases sampling in this build, and the judge's two-sample
vote assumes faithful samples) and disable_eagle_block_drop (every prefix hit otherwise recomputes its last 128-token
block; judging is prefix-heavy). CPU KV offload (64 GiB/rank) keeps shared TB2.1 prefixes warm.

DIR/control.json (tunables over the teacher's, e.g. {"max_batched": 16384}) restarts the engine in place, so prefill
tuning keeps the node; DIR/RELEASE stops the job. DIR/endpoint.json carries the URL once the engine is healthy.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from synth import engine_cache
from synth.common import PROXY_ENV, PYTHON, REPO, atomic, best_effort, read
from synth.condor import requirements
from synth.config import load_teacher
from synth.gpu import assigned, expand, preflight
from synth.probe import free_port, probe, request
from synth.vllm_launch import describe, replica_command, replica_env

SLUG = 'deepseek-v41-flash-1017'
PROTOCOL = {'context_tokens': 132000, 'max_output_tokens': 32768}
SPEC = ('{{"method": "dspark", "num_speculative_tokens": {spec_tokens}, "draft_sample_method": "probabilistic", '
        '"rejection_sample_method": "{rejection}", "enable_adaptive_verification": true, '
        '"disable_eagle_block_drop": {block_drop_off}}}')
DEFAULTS = {'rejection': 'standard', 'block_drop_off': 'true', 'kv_offloading_size': 64, 'prompt_tokens_details': 1,
            'extra_env': {'VLLM_USE_SIMPLE_KV_OFFLOAD': '1'}}
EXCLUDE = ['cpu207', 'cpu307', 'cpu403', 'cpu405', 'gpu132', 'gpu110', 'cpu105', 'gpu180', 'cpu301', 'cpu208']


def teacher():
    spec = load_teacher(SLUG)
    spec = copy.deepcopy(spec)
    spec['serving']['optional_args']['spec_tokens'] = ['--speculative-config', SPEC]
    return spec


def log(home, msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}'
    print(line, flush=True)
    def append():
        with (home / 'events.log').open('a') as f:
            f.write(line + '\n')
    best_effort(append)


def run(home):
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(1))
    spec, host = teacher(), socket.getfqdn()
    scratch = Path(os.environ['_CONDOR_SCRATCH_DIR'])
    local = engine_cache.stable_root(scratch)
    gpus = expand(assigned())
    passed, report = preflight(gpus)
    atomic(home / 'preflight.json', {**report, 'host': host, 'passed': passed})
    if not passed:
        raise RuntimeError(f'GPU preflight failed on {host}')
    gpu = sorted({r.split(',')[2].strip() for r in report['nvidia_smi']['stdout'].splitlines()})
    started = time.time()
    model_dir = local / 'model'
    shutil.copytree(spec['checkpoint'], model_dir, dirs_exist_ok=True)
    log(home, f'staged checkpoint on {host} in {time.time() - started:.0f}s')

    def start(control, n):
        tunables = {**spec['serving']['tunables'], **DEFAULTS, **control}
        port = free_port()
        argv = replica_command(spec, PROTOCOL, port=port, model_dir=str(model_dir), tunables=tunables,
                               rank_ports={'dp_size': len(gpus)})
        engine_scratch = local / 'engines' / 'r0'
        env = replica_env(spec, gpus=gpus, scratch=engine_scratch, tunables=tunables)
        key, facts = engine_cache.config_key(argv, env, spec['runtime']['venv'], gpu)
        seeded = engine_cache.seed(engine_scratch / 'cache', key) if local == engine_cache.STABLE_ROOT else {}
        engine_cache.fresh_vllm_root(env, engine_scratch)
        log_path = engine_scratch / f'vllm-{n}.log'
        atomic(home / f'launch-{n}.json', {**describe(argv, env), 'host': host, 'port': port, 'gpus': gpus,
                                           'control': control, 'engine_cache': seeded, 'time': time.time()})
        proc = subprocess.Popen(argv, env=env, stdout=log_path.open('ab'), stderr=subprocess.STDOUT,
                                start_new_session=True)
        log(home, f'engine {n} started on {host}:{port} (control {control})')
        return {'proc': proc, 'port': port, 'log': log_path, 'n': n, 'started': time.time(), 'healthy': None,
                'key': key, 'facts': facts, 'scratch': engine_scratch, 'control': control}

    def stop(engine):
        proc = engine['proc']
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        best_effort(shutil.copyfile, engine['log'], home / f'vllm-{engine["n"]}.log')

    control = read(home / 'control.json', {})
    engine, n, exits, last_copy = start(control, 0), 0, [], 0
    while not stopping and not (home / 'RELEASE').exists():
        desired = read(home / 'control.json', {})
        if desired != engine['control']:
            log(home, f'control changed {engine["control"]} -> {desired}; restarting')
            stop(engine)
            best_effort((home / 'endpoint.json').unlink, missing_ok=True)
            n += 1
            engine = start(desired, n)
        if engine['proc'].poll() is not None:
            stop(engine)
            exits.append(time.time())
            best_effort((home / 'endpoint.json').unlink, missing_ok=True)
            if len([t for t in exits if time.time() - t < 1800]) >= 3:
                raise RuntimeError('vLLM exited three times within 30 minutes')
            log(home, f'engine {engine["n"]} exited {engine["proc"].returncode}; restarting')
            n += 1
            engine = start(engine['control'], n)
        url = f'http://127.0.0.1:{engine["port"]}/v1'
        if engine['healthy'] is None:
            try:
                request(url + '/models', timeout=3)
                result = probe(url, spec)
                engine['healthy'] = time.time()
                atomic(home / 'endpoint.json', {'url': f'http://{host}:{engine["port"]}/v1', 'host': host,
                       'port': engine['port'], 'served_name': spec['served_name'], 'engine': engine['n'],
                       'control': engine['control'], 'healthy_at': engine['healthy'], 'probe': result,
                       'job_id': os.environ.get('SYNTH_JOB_ID')})
                log(home, f'engine {engine["n"]} healthy after {engine["healthy"] - engine["started"]:.0f}s')
                if local == engine_cache.STABLE_ROOT:
                    best_effort(engine_cache.publish, engine['scratch'] / 'cache', engine['key'], engine['facts'],
                                scratch, {'host': host, 'engine': f'decontam-judge-{engine["n"]}'})
            except OSError:
                pass
            if engine['healthy'] is None and time.time() - engine['started'] > spec['serving']['startup_timeout_sec']:
                raise TimeoutError('vLLM startup exceeded the configured limit')
        if time.time() - last_copy > 60:
            last_copy = time.time()
            best_effort(shutil.copyfile, engine['log'], home / f'vllm-{engine["n"]}.log')
        time.sleep(5)
    log(home, 'RELEASE or signal: stopping')
    stop(engine)
    best_effort((home / 'endpoint.json').unlink, missing_ok=True)


def submit(home, bid=63):
    home.mkdir(parents=True, exist_ok=True)
    env = {'USER': 'user', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONNOUSERSITE': '1',
           'PYTHONPATH': str(REPO), 'SYNTH_JOB_ID': '$(Cluster).$(Process)', **PROXY_ENV}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = home / 'serve.sub'
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = -u -m decontam.serve_judge run --home {home}
initialdir = {REPO}
output = {home}/serve.$(Cluster).out
error = {home}/serve.$(Cluster).err
log = {home}/serve.condor.log
request_cpus = 32
request_memory = {768 * 1024}
request_disk = 1200G
request_gpus = 4
+JobBatchName = "decontam-judge-deepseek"
+MaxRuntime = 1209600
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {requirements(EXCLUDE, True)}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout)
    atomic(home / 'submission.json', {'stdout': r.stdout, 'bid': bid, 'time': time.time(), 'repo': str(REPO)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['run', 'submit'])
    ap.add_argument('--home', type=Path, required=True)
    ap.add_argument('--bid', type=int, default=63)
    a = ap.parse_args()
    submit(a.home, a.bid) if a.action == 'submit' else run(a.home)
