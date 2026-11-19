"""GPU tuning job: measure serving candidates for one teacher on one allocation.

  python -m synth.tune synth/campaigns/<x>.toml --submit    # freeze the tuning campaign, submit the job
  python -m synth.tune                                      # inside the job (SYNTH_CAMPAIGN)

The campaign's [tuning] section names a recorded Pi corpus (synth.bench) and candidates. A
candidate is teacher overrides (deep-merged into the teacher spec, so the winner is a spec edit),
tunables and concurrency levels. The checkpoint is staged once; candidates then run one after
another on the same GPUs through the production launcher and probe (synth.serve.Engine), each
driven by the same replay (fixed levels, or `adapt_hours` for an admission-driven replay written to
ops/tuning/adapt.jsonl). ops/tuning/plan.json is re-read before every candidate, so candidates
can be appended while the job runs. Results: ops/tuning/results.jsonl (candidate x level),
ops/tuning/state.json (per-candidate startup, KV capacity, MoE backend, errors) and engine logs
under ops/tuning/g0/. The job exits when every candidate ran and ops/tuning/DONE exists, or after
`idle_exit_seconds` with nothing to run.
"""
import argparse
import asyncio
import json
from pathlib import Path
import re
import subprocess
import time

from . import condor
from .bench import Replay, build_corpus, levels
from .common import PYTHON, REPO, atomic, read
from .config import Campaign, build, deep_merge
from .gpu import assigned, expand, preflight
from .probe import OPENER
from .serve import Engine, Supervisor, dp_groups, gpu_name
from tbshared import reslog

LOG_FACTS = re.compile(r'KV cache size|Maximum concurrency|MoE backend|Using .*(backend|kernel)|'
                       r'Engram|CUDA out of memory|Error|error', re.I)


class Tuner(Supervisor):
    def __init__(self):
        super().__init__(index=0, generation=0)
        self.home = self.campaign.ops / 'tuning'
        self.home.mkdir(parents=True, exist_ok=True)
        self.base = self.teacher
        self.settings = self.campaign.config['tuning']

    def plan(self):
        plan = read(self.home / 'plan.json')
        if plan is None:
            plan = {'candidates': self.settings['candidates']}
            atomic(self.home / 'plan.json', plan)
        return plan['candidates']

    def wait_gpus_free(self, gpus, timeout=300):
        end = time.time() + timeout
        while time.time() < end:
            out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=60).stdout
            if not any(u in out for u in gpus):
                return True
            time.sleep(5)
        return False

    def candidate(self, k, cand, gpus):
        self.teacher = deep_merge(self.base, cand.get('teacher', {}))
        self.control = {'generation': k, **self.teacher['serving'].get('tunables', {}), **cand.get('tunables', {})}
        serving = self.teacher['serving']
        per = serving.get('gpus_per_replica', 1)
        if serving.get('mode', 'replicas') in ('dp_external_lb', 'dp_internal'):
            groups = dp_groups(gpus, serving)
        else:
            groups = [gpus[i:i + per] for i in range(0, len(gpus), per)]
        # Candidates run one after another in the same per-slot engine dirs as a serving job's engines: they reuse
        # each other's kernels, autotune and compile caches, and seed from (or publish) the same snapshots.
        self.engines = [Engine(self, f'{cand["name"]}-r{i}', g, slot=f'r{i}') for i, g in enumerate(groups)]
        summary = {'name': cand['name'], 'teacher': cand.get('teacher', {}), 'control': self.control}
        started, gates = time.time(), []
        try:
            for engine in self.engines:
                engine.start()
            summary['argv'] = [read(e.run / 'launch.json')['argv'] for e in self.engines]
            summary['engine_cache'] = [e.cache_seed for e in self.engines]
            while not all(e.healthy_at for e in self.engines):
                for engine in self.engines:
                    if not engine.poll():
                        raise RuntimeError(f'{engine.name} exited {engine.proc.returncode} during startup')
                time.sleep(5)
            summary['startup_seconds'] = round(time.time() - started)
            summary['log_facts'] = self.facts()
            targets, metrics = [], []
            for engine in self.engines:
                for rank in engine.ranks:
                    base = engine.urls(rank)[0]
                    size = rank.get('dp_size', 1)
                    targets += ([{'base': base, 'rank': r} for r in range(size)] if size > 1
                                else [{'base': base, 'rank': None}])
                    metrics.append(base.removesuffix('/v1'))
            if cand.get('gate') is not None:
                # Validation arm for the admission gate (aa_evals/qwenserve/gate.py): one gate in front of each
                # engine; agents talk to the gates, throughput is still read from the engines.
                from .serve import consecutive_ports
                for j, target in enumerate(targets):
                    port = consecutive_ports(1)
                    argv = [PYTHON, '-u', '-m', 'aa_evals.qwenserve.gate', '--upstream', target['base'].removesuffix('/v1'),
                            '--port', str(port), '--max-inflight', str(self.control.get('max_seqs', 128)),
                            '--policy', json.dumps(cand['gate'].get('policy', {})),
                            '--log', str(self.home / f'gate-{cand["name"]}-{j}.jsonl'),
                            '--status', str(self.home / f'gate-{cand["name"]}-{j}.status.json'),
                            '--vllm-log', str(self.engines[min(j, len(self.engines) - 1)].local_log)]
                    gates.append(subprocess.Popen(argv, cwd=str(REPO), stdout=(self.home / f'gate-{cand["name"]}-{j}.out').open('ab'),
                                                  stderr=subprocess.STDOUT, start_new_session=True))
                    target['base'] = f'http://127.0.0.1:{port}/v1'
                deadline = time.time() + 120
                while True:
                    try:
                        for target in targets:
                            OPENER.open(target['base'].removesuffix('/v1') + '/gate/status', timeout=5).read()
                        break
                    except OSError:
                        if time.time() > deadline:
                            raise RuntimeError('admission gate did not come up')
                        time.sleep(2)
                summary['gate'] = cand['gate']
            if cand.get('check'):
                # A client check against the healthy engine before the replay (e.g. a vLLM plugin's correctness
                # check): `check` is a script path relative to the repo, run with --base/--model/--corpus/--out.
                out = self.home / f'check-{cand["name"]}.json'
                argv = [PYTHON, str(REPO / cand['check']), '--base', targets[0]['base'].removesuffix('/v1'),
                        '--model', self.teacher['served_name'], '--corpus', str(self.home / 'corpus.json'),
                        '--out', str(out), *map(str, cand.get('check_args', []))]
                r = subprocess.run(argv, capture_output=True, text=True, timeout=cand.get('check_timeout', 3600))
                summary['check'] = {'returncode': r.returncode, 'out': str(out),
                                    'tail': (r.stdout[-3000:] + r.stderr[-2000:])}
            replay = Replay(self.corpus, targets, self.teacher['served_name'], seed=self.settings.get('seed', 0),
                            gap_cap=self.settings.get('gap_cap_seconds', 300),
                            gap_scale=cand.get('gap_scale', self.settings.get('gap_scale', 1.0)),
                            closed_loop=cand.get('closed_loop', self.settings.get('closed_loop', False)),
                            history=cand.get('history', self.settings.get('history', 'inline_think')),
                            stream=cand.get('stream', self.settings.get('stream', True)))
            def healthy():
                """The API process can outlive a dead engine core (it then answers 500s); /health cannot."""
                if any(e.proc.poll() is not None for e in self.engines):
                    return False
                try:
                    for base in metrics:
                        OPENER.open(base + '/health', timeout=30).read()
                    return True
                except OSError:
                    return False
            if cand.get('adapt_hours'):
                # Admission-driven replay (synth.bench adaptive mode): synth.admission picks the concurrency, as a
                # campaign controller does, from this engine's KV geometry and CPU KV tier.
                engine = self.engines[0]
                tokens, block = engine.kv_geometry
                # As serve.records() publishes it to the controller: the backend's KV (per-core x DP ranks), and the
                # summed per-engine gauges divided back to a per-rank mean. dp_external_lb lists one API server per
                # rank (no dp_size): each scrape is already one rank's gauge, but the replay drives all of them.
                dp = max(r.get('dp_size', 1) for r in engine.ranks)
                ranks = dp * len(engine.ranks)
                asyncio.run(replay.adapt(metrics, cand.get('policy', {}), tokens * ranks if tokens else tokens,
                                         hours=cand['adapt_hours'], block=block, kv_divisor=dp, healthy=healthy,
                                         out=self.home / 'adapt.jsonl', tag=cand['name'], offload=engine.kv_offload * ranks))
                summary['adapt'] = {'hours': cand['adapt_hours'], 'policy': cand.get('policy', {}),
                                    'kv_capacity_tokens': tokens, 'kv_offload_tokens': engine.kv_offload}
            else:
                summary['levels'] = asyncio.run(replay.run(
                    levels(cand.get('levels', self.settings['levels'])), metrics,
                    kv_divisor=max(r.get('dp_size', 1) for e in self.engines for r in e.ranks),
                    healthy=healthy, out=self.home / 'results.jsonl', tag=cand['name'],
                    stats_out=self.home / f'turns-{cand["name"]}.jsonl'))
            summary['errors_sample'] = replay.errors[:5]
        except Exception as exc:  # one failed candidate must not end the study
            summary['error'] = repr(exc)[:2000]
            summary['log_facts'] = self.facts()
        finally:
            for g in gates:
                g.terminate()
                try:
                    g.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    g.kill()
            for engine in self.engines:
                engine.stop()
            summary['gpus_released'] = self.wait_gpus_free(gpus)
        return summary

    def facts(self):
        out = []
        for engine in self.engines:
            try:
                lines = engine.local_log.read_text(errors='replace').splitlines()
            except OSError:
                continue
            out += [line[-300:] for line in lines if LOG_FACTS.search(line)][-40:]
        return out

    def main(self):
        import signal
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, 'stopping', True))
        gpus = expand(assigned())
        passed, report = preflight(gpus)
        atomic(self.home / 'preflight.json', {**report, 'host': self.host, 'job_id': self.job_id,
                                              'passed': passed, 'time': time.time()})
        if not passed:
            raise RuntimeError(f'GPU preflight failed on {self.host}')
        self.gpu_name = gpu_name(report, gpus)
        # cgroup CPU/memory every minute (shared memory and the CPU KV tier included), to size each candidate's
        # request from its peak; align with state.json's per-candidate times (#242)
        reslog.start(self.home / f'resources.{self.job_id}.jsonl', 60)   # tbshared's one sampler (#445 item 2)
        # A checkpoint not fetched by synth.fetch (no marker) may be named as already verified by the campaign.
        if not (Path(self.base['checkpoint']) / 'checkpoint-ready.json').exists() and \
                self.settings.get('verified_checkpoint') != self.base['checkpoint']:
            raise RuntimeError(f'{self.base["checkpoint"]} is not a verified checkpoint (synth.fetch)')
        if self.settings.get('corpus_file'):   # a prebuilt corpus (e.g. aa_evals/qwenserve/qa_corpus.py)
            corpus = self.pool.submit(lambda: read(self.settings['corpus_file'])['trajectories'])
        else:
            corpus = self.pool.submit(build_corpus, self.settings['corpus'], self.settings.get('corpus_size', 600),
                                      self.home / 'corpus.json', seed=self.settings.get('corpus_seed', 0))
        self.model_dir = self.stage()  # the corpus index (small Lustre reads) builds meanwhile
        self.corpus = corpus.result()
        self.campaign.log(f'tuning: corpus of {len(self.corpus)} recorded trajectories on {self.host}')
        state = read(self.home / 'state.json', {'done': {}})
        idle_since = None
        while not self.stopping:
            pending = [c for c in self.plan() if c['name'] not in state['done']]
            if not pending:
                if (self.home / 'DONE').exists():
                    break
                idle_since = idle_since or time.time()
                if time.time() - idle_since > self.settings.get('idle_exit_seconds', 1800):
                    break
                time.sleep(15)
                continue
            idle_since = None
            cand = pending[0]
            self.campaign.log(f'tuning: starting {cand["name"]}')
            summary = self.candidate(len(state['done']), cand, gpus)
            state['done'][cand['name']] = summary
            atomic(self.home / 'state.json', state)
            best = max((row.get('srv_gen_tok_s', 0) for row in summary.get('levels', [])), default=None)
            self.campaign.log(f'tuning: {cand["name"]} ' + (f'error {summary["error"][:200]}' if 'error' in summary
                              else f'startup {summary["startup_seconds"]}s, best {best} gen tok/s'))
        self.campaign.log('tuning: finished')


def dry_run(path):
    """Print every candidate's vLLM command (placeholder model dir, port and DP size)."""
    from .vllm_launch import replica_command
    campaign = Campaign(build(path))
    base = campaign.teacher
    for cand in campaign.config['tuning']['candidates']:
        teacher = deep_merge(base, cand.get('teacher', {}))
        tunables = {**teacher['serving'].get('tunables', {}), **cand.get('tunables', {})}
        argv = replica_command(teacher, campaign.protocol, port=8000, model_dir='/model', tunables=tunables,
                               rank_ports={'dp_size': campaign.config['serving']['gpus_per_job'],
                                           'rpc_port': 9000, 'supervisor_port': 9001})
        print(f'{cand["name"]} [{teacher["serving"].get("mode", "replicas")}] {levels(cand.get("levels", campaign.config["tuning"]["levels"]))}')
        print('  ' + ' '.join(argv))


def submit(path, generation=None):
    campaign = Campaign(build(path))
    campaign.freeze()
    if generation is None:
        latest = condor.latest(campaign, 'tuning')
        generation = latest['generation'] + 1 if latest else 0
    s = campaign.config['serving']
    gpus = s['gpus_per_job']
    return condor.submit(campaign, 'tuning', generation=generation, gpus=gpus, cpus=s.get('cpus') or s['cpus_per_gpu'] * gpus,
                         memory_gib=s.get('memory_gib') or s['memory_gib_per_gpu'] * gpus, disk_gib=s['disk_gib'],
                         bid=condor.serving_bid(campaign, gpus),
                         exclude=s.get('exclude_nodes', []) + read(campaign.ops / 'excluded-nodes.json', []),
                         cuda_cap=s.get('cuda_capability', '10.0'))   # 9.0 = H100


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign', nargs='?')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--generation', type=int, help='tuning job generation (default: latest + 1)')
    a = ap.parse_args()
    if a.dry_run:
        dry_run(a.campaign)
    elif a.submit:
        print(submit(a.campaign, a.generation))
    else:
        Tuner().main()
