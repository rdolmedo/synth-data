"""M1 production runs (#118): one command chain that generates M1's data with every decision of
task_synthesis/production/m1.toml, streamed end to end:

  specs (v2h, easy-heavy, one seeded order) ─┬─ eligible share ─► binding (assets.bind run --round, streamed) ─┐
                                             └─ the rest ───────────────────────────────────────────────────┤
  round (one queue, 2 DeepSeek servers as one pool; bound specs: the real-asset preset, others: one-shot) ◄───┘
    ─► task stream (campaign stream --follow) ─► Super campaign ([train.stream], M traces per task, saved-token budget)

  python -m task_synthesis.pipeline.production plan   --name m1-pilot --tokens 1000000000 --milestones 330000000
  python -m task_synthesis.pipeline.production launch --name m1-pilot        # the driver: a CPU job
  python -m task_synthesis.pipeline.production status m1-pilot [--json]          # every component, alerts (#237)
  python -m task_synthesis.pipeline.production replan m1-pilot --vocab v2h0 --seed N [--apply]   # unstarted specs (#238)

plan draws the specs and marks which go to binding (nothing is submitted; check run.json). launch submits the driver
(`drive`), which starts and watches everything and is safe to restart: the generator servers, the bind job, the round
jobs, the stream producer, and the Super campaign once the stream's first chunk exists. It stops the round (<round>/STOP)
as soon as the tasks streamed or accepted cover the token budget, releases the generator servers when the round has
finished (no switch to Super, #137), records when the campaign's saved tokens pass each milestone (e.g. the pilot's
330M point, #137), and exits when the campaign is complete. State: <ROOT>/production/<name>/ (run.json, state.json,
driver.log, campaign.toml).
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import tomllib
import zlib
from pathlib import Path

from ..common import REPO, ROOT, frozen_checkout
from tbshared import serving   # the registry reader (#445 item 3)

RUNS = ROOT / 'production'
ROUNDS = ROOT / 'rounds'
STREAMS = ROOT / 'streams'
SERVING = None   # the registry root: serving_dir() ($TASKSYNTH_SERVING_ROOT, else production; a test sets a Path)
BINDS = ROOT / 'assets' / 'binds'
SYNTH = Path('/data/aai/tbench/synth')
CONFIG = REPO / 'task_synthesis' / 'production' / 'm1.toml'
HPY = '/opt/aai/venvs/harbor/bin/python'
APY = '/data/aai/venvs/tasksynth-assets/bin/python'
# the round jobs' exclusions (round.submit) plus /data write problems (gpu123 lost writes 11-01, gpu143 broken I/O)
BAD_CPU_NODES = ('gpu110', 'gpu132', 'gpu180', 'cpu105', 'cpu301', 'gpu158', 'cpu208', 'gpu109', 'gpu123', 'gpu143')


# /data's per-user FILE quota (41.9M files) is the limit that bites: at the limit every new file fails with EDQUOT,
# and on 2026-10-25 a 4xB200 server and ~100 workers ran for nothing. A round's outputs are loose files (#124 pilot:
# ~38 per accepted task: tasks/ ~27, work/ ~9, assets/ ~2); the Super campaign packs its attempts as it runs.
# `plan`, `launch` and the driver only warn (user, 2026-11-04 and 2026-11-05: no refusal at launch either; #245, R10). Finished rounds are packed with task_synthesis.pipeline.roundpack.
QUOTA_DEFAULTS = {'files_per_task': 40, 'campaign_loose_files': 200_000, 'alert_free_files': 2_000_000}


def fast_files(path='/data'):
    """(files used, files limit) of this user's quota on `path`, or None when lfs is unavailable/unparseable."""
    try:
        r = subprocess.run(['lfs', 'quota', '-u', os.environ.get('USER') or str(os.getuid()), path],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == path:
            try:
                used, limit = int(f[5].rstrip('*')), int(f[7])
            except ValueError:
                return None
            return (used, limit) if limit > 0 else None
    return None


def quota_cfg(cfg):
    return {**QUOTA_DEFAULTS, **cfg.get('quota', {})}


# ------------------------------------------------------------------------------------------------------------- plan
def eligible(spec, excluded):
    """A spec that can go to binding: a generated anchor (bucket best-generated) of a class binding serves."""
    if spec.get('bucket') != 'best-generated' or not spec.get('assets'):
        return False
    anchor = next((x for x in spec['assets'] if x.get('role') == 'anchor'), None)
    return anchor is not None and anchor['class'] not in excluded


def sizing(cfg, budget, eligible_share):
    """Expected Super saved tokens per spec and the specs the budget needs (an estimate: the driver stops the round
    once the stream covers the budget)."""
    rec, sp = cfg['recipe'], cfg['specs']
    share = rec['asset_share'] if rec['real_assets'] != 'off' else 0.0
    p_asset = share * eligible_share * sp['bind_rate']
    per_trace = p_asset * sp['tokens_per_task_asset'] + (1 - p_asset) * sp['tokens_per_task_generated']
    per_task = rec['traces_per_task'] * per_trace
    tasks = budget / per_task
    return {'p_asset_tasks': round(p_asset, 4),
            'asset_token_share': round(p_asset * sp['tokens_per_task_asset'] / per_trace, 4),
            'tokens_per_trace_prior': round(per_trace),
            'tasks_needed': math.ceil(tasks), 'specs_sized': math.ceil(tasks * sp['oversupply'])}


def run_seed(cfg, name):
    return cfg['specs']['seed'] * 1000 + zlib.crc32(name.encode()) % 1000


def seed_in_use(seed):
    """Spec ids are s<seed>-NNNNN: two rounds drawn with the same seed would hold the same specs."""
    for f in ROUNDS.glob('*/specs.jsonl'):
        try:
            with open(f) as fh:
                first = fh.readline()
            if first and json.loads(first).get('spec_id', '').startswith(f's{seed}-'):
                return f.parent.name
        except (OSError, ValueError):
            continue
    return None


def plan(a):
    from . import specs as specs_mod
    from .round import REAL_ASSET_PRESETS
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,40}', a.name):
        raise SystemExit('--name: lowercase letters, digits and dashes')
    cfg_path = Path(a.config).resolve()
    cfg = tomllib.loads(cfg_path.read_text())
    rec = cfg['recipe']
    if rec['real_assets'] not in REAL_ASSET_PRESETS:
        raise SystemExit(f'real_assets {rec["real_assets"]!r}: one of {sorted(REAL_ASSET_PRESETS)}')
    if rec['vocab'] not in specs_mod.VOCAB_NAMES:
        raise SystemExit(f'vocab {rec["vocab"]!r}: one of {specs_mod.VOCAB_NAMES}')
    mix = specs_mod.parse_mix(rec['minutes_mix'])
    rd, run = ROUNDS / a.name, RUNS / a.name
    for p in (run / 'run.json', rd / 'specs.jsonl', STREAMS / a.name):
        if p.exists():
            raise SystemExit(f'{p} exists: pick a new run name')
    seed = a.seed if a.seed is not None else run_seed(cfg, a.name)
    if (other := seed_in_use(seed)):
        raise SystemExit(f'seed {seed} already drew round {other}: pass another --seed')
    excluded = set(rec.get('bind_excluded_classes', []))
    first = sizing(cfg, a.tokens, 0.70)            # eligible share guess for the first draw; measured below
    n = a.specs or first['specs_sized']
    print(f'drawing {n} specs (seed {seed}, vocab {rec["vocab"]}, minutes {rec["minutes_mix"]})', flush=True)
    drawn = specs_mod.sample(n, seed, None, mix, rec['vocab'])
    elig = [s for s in drawn if eligible(s, excluded)]
    share = rec['asset_share'] if rec['real_assets'] != 'off' else 0.0
    chosen = set(s['spec_id'] for s in random.Random(f'{seed}-bind').sample(elig, round(share * len(elig))))
    for s in drawn:
        if s['spec_id'] in chosen:
            s['bind'] = 'pending'
    size = sizing(cfg, a.tokens, len(elig) / len(drawn))
    if not a.specs and size['specs_sized'] > len(drawn):
        # the measured eligible share needs more specs than the first guess: draw again at the right size
        drawn = specs_mod.sample(size['specs_sized'], seed, None, mix, rec['vocab'])
        elig = [s for s in drawn if eligible(s, excluded)]
        chosen = set(s['spec_id'] for s in random.Random(f'{seed}-bind').sample(elig, round(share * len(elig))))
        for s in drawn:
            if s['spec_id'] in chosen:
                s['bind'] = 'pending'
        size = sizing(cfg, a.tokens, len(elig) / len(drawn))
    servers = a.servers.split('+') if a.servers else [f'{a.name}-ds{i + 1}' for i in range(cfg['generator']['servers'])]
    info = {
        'name': a.name, 'created': time.time(), 'config_path': str(cfg_path), 'config': cfg,
        'commit': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'tokens_budget': a.tokens, 'milestones': sorted(a.milestones or []), 'seed': seed,
        'eligible_share': round(len(elig) / len(drawn), 4), **size,
        'specs': len(drawn), 'eligible': len(elig), 'bind_selected': len(chosen),
        'round': a.name, 'stream': str(STREAMS / a.name), 'bind': a.name,
        'campaign_id': f'{time.strftime("%Y%m%d")}-nemotron35-super-tasksynth-{a.name}',
        'servers': servers, 'own_servers': not a.servers,
        # one spare serving name per own server: the driver swaps a server before its ~24 h limit by starting the
        # spare, then releasing the old one once the spare is healthy; both names are in the round's pool
        'server_spares': {s: f'{s}b' for s in servers} if not a.servers else {},
        # the registry the run's servers, driver and jobs use, only when not production (run.json unchanged then)
        **({'serving_env': serving.to_record()} if serving.to_record() else {}),
    }
    q, files = quota_cfg(cfg), fast_files()
    need = int(info['tasks_needed'] * q['files_per_task'] + q['campaign_loose_files'])
    info['files_estimate'] = need
    if files:
        free = files[1] - files[0]
        print(f'/data files: {files[0]:,} of {files[1]:,} used, {free:,} free; this run needs ~{need:,} '
              f'({info["tasks_needed"]:,} tasks x {q["files_per_task"]} + {q["campaign_loose_files"]:,})', flush=True)
        if free - need < q['alert_free_files']:
            print(f'WARNING /data file quota: {free:,} free - ~{need:,} for this run leaves less than '
                  f'{q["alert_free_files"]:,}; free files (pack/delete old rounds and campaigns) before it runs out',
                  flush=True)
    if a.dry_run:
        print(json.dumps({k: v for k, v in info.items() if k != 'config'}, indent=1))
        return
    rd.mkdir(parents=True, exist_ok=True)
    tmp = rd / '.specs.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(s) + '\n' for s in drawn))
    os.replace(tmp, rd / 'specs.jsonl')
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'run.json', info)
    print(json.dumps({k: v for k, v in info.items() if k != 'config'}, indent=1))
    print(f'{run}/run.json; next: python -m task_synthesis.pipeline.production launch --name {a.name}')


# ------------------------------------------------------------------------------------------------------------ helpers
def autoscale_toml(cfg):
    """[campaign.autoscale] of the run's config -> the body of the campaign's [serving.autoscale] table (#234 R1:
    Super starts with `start` engines and follows demand up to [campaign] engines; synth/autoscale.py). Runs planned
    before it have none: autoscale off, every engine at once as before."""
    if not cfg:
        return 'enabled = false'
    lines = []
    for k, v in cfg.items():
        if not re.fullmatch(r'[a-z_]+', k) or isinstance(v, (dict, list, str)):
            raise SystemExit(f'[campaign.autoscale] {k} = {v!r}: numbers and booleans only')
        lines.append(f'{k} = {json.dumps(v)}')
    return '\n'.join(lines)


def serving_dir() -> Path:
    """The serving registry this process reads: SERVING when set (tests), else tbshared.serving.registry_root(), the
    one serve_teacher writes ($TASKSYNTH_SERVING_ROOT, e.g. a canary, else production). A Driver pins it to its run's."""
    return serving.registry_root(SERVING)


def run_root_env(info) -> dict:
    """The registry root a run was planned under, validated, as serving.forwarded() gives it: run.json `serving_env`,
    recorded by `plan` only for a registry other than production ({} = production, as for every run planned before)."""
    return serving.forwarded(info.get('serving_env') or {})


def pin_run_root(info) -> dict:
    """This process onto the run's registry (tbshared.serving.pin): every reader here and every command and job it
    starts (they inherit it, or get it forwarded). Refused when $TASKSYNTH_SERVING_ROOT names another registry."""
    return serving.pin(info.get('serving_env'), f'run {info.get("name")}')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def job_states(job_id):
    """Condor JobStatus of each process of a cluster (1 idle, 2 running, 5 held; absent: gone)."""
    cluster = str(job_id).split('.')[0]
    r = subprocess.run(['condor_q', cluster, '-af', 'ProcId', 'JobStatus'], capture_output=True, text=True)
    if r.returncode != 0:
        return None   # schedd unreachable: unknown
    return {int(p): int(s) for p, s in (l.split() for l in r.stdout.splitlines() if l.strip())}


def alive(job_id):
    st = job_states(job_id)
    return None if st is None else any(s in (1, 2, 5) for s in st.values())


def round_finished(rd):
    from .roundpack import RoundFiles
    sids = [json.loads(l)['spec_id'] for l in (rd / 'specs.jsonl').read_text().splitlines() if l.strip()]
    return RoundFiles(rd).has_outcome(sids)


def healthy(server):
    """The server's endpoints.json is fresh and healthy (tbshared.serving.endpoint)."""
    return serving.endpoint(server, serving_dir()) is not None


# ---------------------------------------------------------------------------------------------------------- driver
class Driver:
    def __init__(self, name):
        self.run = RUNS / name
        self.info = read_json(self.run / 'run.json')
        if not self.info:
            raise SystemExit(f'{self.run}/run.json missing: run `plan` first')
        self.cfg = self.info['config']
        self.name = name
        self.rd = ROUNDS / self.info['round']
        self.stream = Path(self.info['stream'])
        self.camp = SYNTH / self.info['campaign_id']
        self.code = REPO
        self.state_path = self.run / 'state.json'
        self.state = read_json(self.state_path, {})
        self.spares = self.info.get('server_spares') or {}       # runs planned before #124 have none
        pin_run_root(self.info)   # this process, its commands and its jobs: the run's registry
        self.pool = '+'.join(self.info['servers'] + [self.spares[x] for x in self.info['servers'] if x in self.spares])
        self.records = {}   # spec id -> accepted (valid and judge-passed); records are final once written

    def save(self):
        write_json(self.state_path, self.state)

    def log(self, msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        with open(self.run / 'driver.log', 'a') as f:
            f.write(line + '\n')

    def sh(self, argv, what):
        r = subprocess.run(argv, cwd=self.code, capture_output=True, text=True,
                           env={**os.environ, 'PYTHONPATH': str(self.code), 'PYTHONNOUSERSITE': '1'})
        if r.returncode != 0:
            self.log(f'ALERT {what} failed (rc {r.returncode}): {(r.stderr or r.stdout)[-1500:]}')
            return None
        return r.stdout

    # ---- generator servers
    def server_names(self):
        """Every serving name of the run's own servers (each slot's primary and its spare)."""
        return [n for s in self.info['servers'] for n in ([s, self.spares[s]] if s in self.spares else [s])]

    def submit_server(self, name):
        g = self.cfg['generator']
        # --pipeline task-synthesis: this pipeline's server profiles and submit defaults in the one launcher (#480)
        argv = [HPY, '-m', 'task_synthesis.serve.serve_teacher', 'submit', '--pipeline', 'task-synthesis',
                '--teacher', g['teacher'], '--name', name,
                '--gpus', str(g['gpus']), '--bid', str(g['bid']), '--hours', str(g['hours'])]
        if read_json(serving_dir() / name / 'job.json'):
            argv.append('--force')
        out = self.sh(argv, f'server {name} submit')
        if out:
            self.state.setdefault('server_jobs', {}).setdefault(name, []).append(out.strip().splitlines()[-1])
            self.save()
            self.log(f'server {name}: job {out.strip().splitlines()[-1]}')
        return out

    def held_server_removed(self, name, job):
        """A held server never blocks the run (#378 decision, tbshared.holds): alive() counts a held job as alive, so
        before this a held DeepSeek server was never replaced. Every hold but a user's own (task-synthesis serve_teacher
        submits without the memory net, so nothing releases a memory hold either; it gets the shared net and the node
        exclusion with the shared serving harness) is removed while still held, at most holds.MAX_REPLACEMENTS times
        per server, then ALERTed. Used for a slot's server, a swap's new server and the server kept for decontam.
        Returns True when it was removed (the caller restarts the server)."""
        from tbshared import holds
        st = job_states(job['job_id'])
        if not st or not any(v == 5 for v in st.values()):
            return False
        n = self.state.setdefault('held_replacements', {}).get(name, 0)
        if n >= holds.MAX_REPLACEMENTS:
            if not self.state.get(f'held_alerted_{name}'):
                self.state[f'held_alerted_{name}'] = time.time()
                self.save()
                self.log(f'ALERT server {name} held again after {n} replacements: left held (operator check)')
            return False
        cluster = str(job['job_id']).split('.')[0]
        removed = False
        for proc, status in sorted(st.items()):
            if status == 5:
                # serve_teacher here submits without the memory net: a memory hold is never released either
                action, node, code = holds.remove_held(f'{cluster}.{proc}', has_net=False)
                if action == 'replace':
                    removed = True
                    self.log(f'server {name}: job {cluster}.{proc} held (code {code}) on {node}: removed, restarting')
        if removed:
            self.state['held_replacements'][name] = n + 1
            self.save()
        return removed

    def remove_server(self, name, why):
        """-> True once the server's job is gone (removed now or already), False if that could not be established
        (condor_q failing, e.g. the scheduler down at 2026-11-05 07:34): the caller retries on its next pass."""
        job = read_json(serving_dir() / name / 'job.json')
        if not job:
            return True
        q = subprocess.run(['condor_q', job['job_id'], '-af', 'Args'], capture_output=True, text=True)
        if q.returncode != 0 or 'Error' in q.stderr or 'SCHEDD' in q.stderr:
            self.log(f'server {name}: condor_q failed ({(q.stderr or "").strip()[:120]}); release retried next pass')
            return False
        if not q.stdout.strip():
            return True   # the job is no longer in the queue
        if re.search(rf'--name {re.escape(name)}(\s|$)', q.stdout):   # never remove a job that is not this server
            r = subprocess.run(['condor_rm', job['job_id']], capture_output=True, text=True)
            if r.returncode != 0:
                self.log(f'server {name}: condor_rm failed ({(r.stderr or "").strip()[:120]}); retried next pass')
                return False
            self.log(f'released server {name} (job {job["job_id"]}): {why}')
        return True

    def servers_up(self):
        """Keep one live server per slot. A slot's server older than [generator] hours - swap_lead_hours is swapped:
        the slot's other name (primary <-> spare) is submitted first, and the old server is released once the new one
        is healthy (users' rule: no server past ~24 h; submit the replacement first). The round's pool lists both names
        and re-reads its members every 5 min, so sessions move over without a restart (a request cut by the release is
        retried on a live member)."""
        if not self.info['own_servers'] or self.state.get('servers_released') or (self.rd / 'STOP').exists():
            return
        g = self.cfg['generator']
        lead = g.get('swap_lead_hours', 1.0)
        active = self.state.setdefault('server_active', {})
        swaps = self.state.setdefault('server_swaps', {})
        for slot in self.info['servers']:
            cur = active.get(slot, slot)
            job = read_json(serving_dir() / cur / 'job.json')
            sw = swaps.get(slot)
            if sw:   # a swap in flight: release the old server once the new one serves
                if healthy(sw['new']):
                    if not self.remove_server(cur, f'swapped for {sw["new"]} (healthy after '
                                                   f'{(time.time() - sw["time"]) / 60:.0f} min)'):
                        continue   # the old server's removal is not confirmed yet: retried next pass
                    active[slot] = sw['new']
                    swaps.pop(slot)
                    self.save()
                    continue
                new_job = read_json(serving_dir() / sw['new'] / 'job.json')
                if new_job and (alive(new_job['job_id']) is False or self.held_server_removed(sw['new'], new_job)):
                    self.log(f'ALERT server {sw["new"]} (swap for {cur}) is gone before it was healthy; resubmitting')
                    self.submit_server(sw['new'])
                    sw['time'] = time.time()
                    self.save()
                elif time.time() - sw['time'] > 2 * 3600 and not sw.get('alerted'):
                    sw['alerted'] = True
                    self.save()
                    self.log(f'ALERT server {sw["new"]} (swap for {cur}) not healthy after 2 h; {cur} keeps serving')
                continue   # the old server keeps serving (or, if it died, the spare is already on its way)
            if job and self.held_server_removed(cur, job):
                job = None   # removed: restarted below under its name
            if job and alive(job['job_id']) is not False:
                if slot in self.spares and time.time() - job.get('time', time.time()) > (g['hours'] - lead) * 3600:
                    new = self.spares[slot] if cur == slot else slot
                    if self.submit_server(new):
                        swaps[slot] = {'new': new, 'old': cur, 'time': time.time()}
                        self.save()
                        self.log(f'server swap {cur} -> {new}: {cur} is {(time.time() - job["time"]) / 3600:.1f} h old')
                continue
            if self.submit_server(cur):   # dead (or never started): restart it under its name
                swaps.pop(slot, None)
                self.save()

    def decontam_hold(self):
        """#127: `decontam.pool run --production <run.json> --follow` writes <run>/decontam.HOLD while it follows the run
        and removes it after its last pass (at stream END), which judges on these DeepSeek servers. Keep them while the
        file exists, its job (CONDOR_JOB_ID in the file) is alive and the round finished less than [decontam] hold_hours
        (3) ago; then release them with an ALERT."""
        hold = self.run / 'decontam.HOLD'
        if not hold.exists() or self.state.get('hold_dropped'):
            return False
        info = read_json(hold, {}) or {}
        why = None
        if info.get('job') and alive(info['job']) is False:
            why = f'its job {info["job"]} is gone'
        limit = (self.cfg.get('decontam') or {}).get('hold_hours', 3)
        since = self.state.get('round_finished')
        if since and time.time() - since > limit * 3600:
            why = f'held {limit} h past the round\'s end'
        if why:
            self.state['hold_dropped'] = time.time()
            self.save()
            self.log(f'ALERT decontam hold {hold}: {why}; releasing the servers')
            return False
        if since and not self.state.get('hold_logged'):
            self.state['hold_logged'] = time.time()
            self.save()
            self.log(f'servers kept for the decontam last pass ({hold}: job {info.get("job")}, work {info.get("work")})')
        return True

    def decontam_keep(self):
        """The server decontam keeps from the round's STOP (#234 R2): the one the HOLD file names ("keep"), else the
        one carrying the most judge load (the judge's --load-file under the HOLD's work dir, fresh within 15 min:
        groups pinned per target, mapped to servers by their endpoints.json urls), else the first healthy one."""
        names = self.server_names()
        info = read_json(self.run / 'decontam.HOLD', {}) or {}
        if info.get('keep') in names:
            return info['keep'], 'named in the HOLD file'
        urls = {}
        for n in names:
            for u in (read_json(serving_dir() / n / 'endpoints.json', {}) or {}).get('urls', []):
                urls[u.rstrip('/')] = n
        load = {}
        if info.get('work'):
            for f in sorted((Path(info['work']) / 'judge-load').glob('*.json')):
                rec = read_json(f, {}) or {}
                if time.time() - rec.get('updated_at', 0) > 900:
                    continue
                for url, _rank, weight in rec.get('targets', []):
                    if (n := urls.get(str(url).rstrip('/'))):
                        load[n] = load.get(n, 0) + weight
        if load:
            best = max(names, key=lambda n: (load.get(n, 0), -names.index(n)))
            return best, f'most judge load ({load})'
        live = [n for n in names if healthy(n)]
        return (live or names)[0], 'first healthy server (no judge load reported)'

    def decontam_trim(self):
        """#234 R2: at the round's STOP, while a decontam hold exists, keep one server for decontam and release the
        others (the round's in-flight specs at STOP are surplus; m1-hero kept both servers 80 min for ~1 server of judge
        load). The judge re-pins the released servers' groups. The kept one is released when the hold goes (or at
        round finish without a hold, as before)."""
        if (not self.info['own_servers'] or self.state.get('servers_released') or not (self.rd / 'STOP').exists()
                or not self.decontam_hold()):
            return
        kept = self.state.get('decontam_kept')
        if kept is None:
            keep, why = self.decontam_keep()
            kept = self.state['decontam_kept'] = {'server': keep, 'why': why, 'time': time.time(), 'released': []}
            self.save()
            self.log(f'round STOP with a decontam hold: keeping {keep} for decontam ({why}); releasing the others')
        left = [n for n in self.server_names() if n != kept['server'] and n not in kept['released']]
        for n in left:
            if self.remove_server(n, f'round STOP: decontam keeps {kept["server"]}'):
                kept['released'].append(n)
                self.save()
        job = read_json(serving_dir() / kept['server'] / 'job.json')
        if job and kept.get('restarts', 0) < 2 and (alive(job['job_id']) is False
                                                    or self.held_server_removed(kept['server'], job)):
            kept['restarts'] = kept.get('restarts', 0) + 1
            self.save()
            self.log(f'ALERT the server kept for decontam ({kept["server"]}) is gone; resubmitting '
                     f'({kept["restarts"]} of 2)')
            self.submit_server(kept['server'])

    def hold_just_released(self):
        """Servers are kept only for a decontam hold that no longer exists: release them now, not at the next pass."""
        return (bool(self.state.get('decontam_kept') or self.state.get('hold_logged'))
                and not self.state.get('servers_released') and not (self.run / 'decontam.HOLD').exists())

    def nap(self, seconds, poll=30):
        """Sleep until the next pass, waking early once a decontam hold is gone (#234 R2: release within ~1 min)."""
        end = time.time() + seconds
        while (left := end - time.time()) > 0:
            time.sleep(min(poll, left))
            if self.hold_just_released() and not getattr(self, '_woke_for_release', False):
                self._woke_for_release = True   # once: a release that fails (condor_q down) waits for the next pass
                return

    def release_servers(self, why):
        if not self.info['own_servers'] or self.state.get('servers_released'):
            return
        done = [self.remove_server(s, why) for s in self.server_names()]
        if all(done):
            self.state['servers_released'] = time.time()
            self.save()

    # ---- bind job
    def bind_done(self):
        if not self.info['bind_selected']:
            return True
        pending = [json.loads(l)['spec_id'] for l in (self.rd / 'specs.jsonl').read_text().splitlines()
                   if '"bind": "pending"' in l]
        have = set(os.listdir(self.rd / 'bind')) if (self.rd / 'bind').exists() else set()
        return all(f'{s}.json' in have for s in pending)

    def servers_ready(self):
        """Round shards and the bind job start only once a generator server is healthy (#378 port of eval #263): they
        only wait for the servers before that (round.py wait_healthy, up to 3 h), and 12 shards x 24 units idling
        through a 4xB200 server's queue wait and start-up cost ~1-36k credits per launch (c30: DS waited 5.1 h for
        nodes). Once started they are never held back again (server swaps, restarts). ALERT after [round]
        wait_alert_hours (3) of waiting. [round] wait_for_server = false restores the old behaviour."""
        if (self.state.get('dependents_started') or (self.rd / 'job.json').exists()   # a round already submitted
                or not self.cfg.get('round', {}).get('wait_for_server', True)):
            return True
        up = [s for s in self.server_names() if healthy(s)]   # any pool member (a slot's spare too, after a swap)
        if not up:
            if not self.state.get('dependents_waiting'):
                self.state['dependents_waiting'] = time.time()
                self.log('round and bind wait for a healthy generator server')
                self.save()
            hours = self.cfg.get('round', {}).get('wait_alert_hours', 3)   # round shards used to give up after 3 h
            waited = time.time() - self.state['dependents_waiting']
            if waited > hours * 3600 and not self.state.get('dependents_alerted'):
                self.state['dependents_alerted'] = time.time()
                self.log(f'ALERT no generator server healthy after {waited / 3600:.1f} h: the round and bind are '
                         f'still not submitted (check the servers: `production status`)')
                self.save()
            return False
        waited = time.time() - self.state.get('dependents_waiting', time.time())
        self.state['dependents_started'] = time.time()
        self.log(f'server {up[0]} healthy: starting the round and bind (waited {waited / 60:.0f} min)')
        self.save()
        return True

    def bind_up(self):
        if not self.info['bind_selected'] or self.state.get('bind_done'):
            return
        if self.bind_done() or (self.rd / 'STOP').exists() or self.state.get('round_finished'):
            self.state['bind_done'] = time.time()
            self.save()
            return
        job = read_json(BINDS / self.info['bind'] / 'job.json')
        if job and (job.get('args') or {}).get('round') == self.info['round'] and alive(job['job_id']) is not False:
            return
        n = self.state.get('bind_submits', 0)
        if n >= 6:
            if not self.state.get('bind_alerted'):
                self.log('ALERT bind job resubmitted 6 times; not again (round waits on --bind-window)')
                self.state['bind_alerted'] = True
            return
        # the first bind job starts with every slot open: at launch the generator servers are idle and binding is the
        # bottleneck (#172 entry 3); a resubmitted one ramps its slots into an already loaded pool (entry 11, #238)
        out = self.sh(self.bind_args() + (['--ramp-minutes', '0'] if n == 0 else []), 'bind submit')
        if out:
            self.state['bind_submits'] = n + 1
            self.log(f'bind job {out.strip().splitlines()[-1]} (submit {n + 1})')
            self.save()

    def bind_args(self):
        """[bind] concurrency (flat), or concurrency_per_server x the run's servers when set (spares not counted)."""
        b, n = self.cfg['bind'], len(self.info['servers'])
        slots = b['concurrency_per_server'] * n if b.get('concurrency_per_server') else b['concurrency']
        http = b['http_concurrency_per_server'] * n if b.get('http_concurrency_per_server') else b['http_concurrency']
        return [APY, '-m', 'task_synthesis.assets.bind', 'submit', '--name', self.info['bind'], '--round', self.info['round'],
                '--llm', self.pool, '--concurrency', str(slots), '--http-concurrency', str(http),
                '--warm-from', b.get('warm_from', 'none'), '--grow-now', str(b.get('grow_now', 0)),
                '--cpus', str(b['cpus']), '--memory-gib', str(b['memory_gib']), '--bid', str(b['bid'])]

    # ---- round
    def round_args(self):
        r, rec = self.cfg['round'], self.cfg['recipe']
        oneshot = rec['real_assets'] in ('oneshot', 'off')
        argv = [HPY, '-m', 'task_synthesis.pipeline.round', 'submit', '--round', self.info['round'],
                '--n', str(self.info['specs']), '--seed', str(self.info['seed']), '--gen', self.pool, '--judge', self.pool,
                '--real-assets', rec['real_assets'], '--vocab', rec['vocab'], '--minutes-mix', rec['minutes_mix'],
                '--jobs', str(r['jobs']), '--authors', str(0 if oneshot else r['authors']),
                '--cpus', str(r['cpus'] if oneshot else r.get('cpus_workspace', r['cpus'])),   # #242
                '--memory-gib', str(r['memory_gib']), '--disk-gib', str(r['disk_gib']),
                '--build-parallel', str(r['build_parallel']),
                '--target-inflight', str(r['oneshot_target_inflight'] if oneshot else r['target_inflight']),
                '--target-sessions', str(r['target_sessions']), '--bind-window', str(r['bind_window']),
                '--bid', str(r['bid'])] + (['--cheap-bid', str(r['cheap_bid'])] if r.get('cheap_bid') else [])
        if oneshot:
            # one-shot specs are admitted by the pool's load (round.admit); --concurrency is only each job's ceiling,
            # twice its share of the servers' knee, so the other jobs can fill in for a lost one
            argv += ['--concurrency',
                     str(math.ceil(2 * r['knee_inflight_per_server'] * len(self.info['servers']) / r['jobs']))]
        return argv

    def round_up(self):
        job = read_json(self.rd / 'job.json')
        if job:
            self.rebid_round(job)
            if (not self.state.get('round_finished') and alive(job['job_id']) is False and not self.state.get('round_alerted')
                    and not round_finished(self.rd)):   # jobs exit by themselves once every spec has its record
                self.log('ALERT every round job is gone but the round has not finished (open claims of the lost jobs '
                         'are taken over by any new job of the round: check round.log, then `round submit` again by hand)')
                self.state['round_alerted'] = True
            return
        out = self.sh(self.round_args(), 'round submit')
        if out:
            self.log(f'round {self.info["round"]}: job {out.strip().splitlines()[-1]} ({self.cfg["round"]["jobs"]} jobs)')

    def rebid_round(self, job):
        """Round jobs submitted at --cheap-bid that are still idle after [round] cheap_bid_wait_min (5): raise them to
        the round's bid in place, so a cheap node taken since the capacity check does not hold a shard back (#242)."""
        args, r = job.get('args') or {}, self.cfg['round']
        if not job.get('cheap_jobs') or job.get('rebid_at') or not args.get('cheap_bid'):
            return
        if time.time() - job['time'] < 60 * r.get('cheap_bid_wait_min', 5):
            return
        cluster = str(job['job_id']).split('.')[0]
        r2 = subprocess.run(['condor_q', cluster, '-af', 'ProcId', 'JobStatus', 'JobPrio'], capture_output=True, text=True)
        if r2.returncode != 0:
            return
        idle = [p for p, s, prio in (l.split() for l in r2.stdout.splitlines() if l.strip())
                if s == '1' and 1000 + int(prio) < args['bid']]
        for p in idle:
            subprocess.run(['condor_qedit', f'{cluster}.{p}', 'JobPrio', str(args['bid'] - 1000)], capture_output=True,
                           text=True, timeout=60)
        if idle:
            self.log(f'round {self.info["round"]}: {len(idle)} jobs idle at bid {args["cheap_bid"]} after '
                     f'{r.get("cheap_bid_wait_min", 5)} min: rebid to {args["bid"]}')
        job['rebid_at'] = time.time()
        write_json(self.rd / 'job.json', job)

    # ---- stream producer
    def producer_up(self):
        if (self.stream / 'END.json').exists():
            return
        job = self.state.get('producer_job')
        if job and alive(job) is not False:
            return
        rounds = self.info['round']
        sub = self.run / 'producer.sub'
        sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'cd {self.code} && PYTHONPATH={self.code} PYTHONNOUSERSITE=1 exec {HPY} -u -m task_synthesis.pipeline.campaign stream --rounds {rounds} --stream {self.stream} --follow --interval 300 --min-tasks 50 --end'"
output = {self.run}/producer.$(Cluster).out
error = {self.run}/producer.$(Cluster).err
log = {self.run}/producer.condor.log
request_cpus = 4
request_memory = 32768
request_disk = 20G
+JobBatchName = "tasksynth-{self.name}-producer"
+MaxRuntime = 604800
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin{serving.condor_env(run_root_env(self.info))}"
requirements = {' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in BAD_CPU_NODES)}
queue
''')
        out = self.sh(['condor_submit_bid', '35', str(sub)], 'producer submit')
        if out and (m := re.search(r'cluster (\d+)', out)):
            self.state['producer_job'] = f'{m.group(1)}.0'
            self.save()
            self.log(f'stream producer job {m.group(1)}.0')

    # ---- campaign
    def campaign_toml(self, smoke):
        c, rec = self.cfg['campaign'], self.cfg['recipe']
        text = (self.code / c['template']).read_text()
        # one uncapped smoke attempt (user, 2026-11-04: minimal, check in production): the capped one (128-token replies
        # to a 32k context) ran 26.5 min on the m1-pilot while the other engines waited (#124)
        att = '\n'.join(f'  {{ task = "{t}", kind = "train" }}' for t in smoke[:1])
        subs = {'@ID@': self.info['campaign_id'], '@STREAM@': str(self.stream), '@SMOKE@': att,
                '@TRACES_PER_TASK@': str(rec['traces_per_task']), '@TOKENS_BUDGET@': str(self.info['tokens_budget']),
                '@TOKENS_PRIOR@': str(self.info['tokens_per_trace_prior']), '@ENGINES@': str(c['engines']),
                '@SERVING_BID@': str(c['serving_bid']), '@WORKERS@': str(c['workers']),
                '@MAX_WORKERS@': str(c['max_workers']), '@AUTOSCALE@': autoscale_toml(c.get('autoscale'))}
        for k, v in subs.items():
            text = text.replace(k, v)
        if re.search(r'@[A-Z_]+@', text):
            raise SystemExit(f'campaign template: unfilled markers {re.findall(r"@[A-Z_]+@", text)}')
        tomllib.loads(text)
        return text

    def campaign_up(self):
        if (self.camp / 'ops' / 'campaign.json').exists() or self.state.get('campaign_failed'):
            return
        smoke = read_json(self.stream / 'smoke.json')
        if not smoke:
            return
        path = self.run / 'campaign.toml'
        path.write_text(self.campaign_toml(smoke))
        out = self.sh([HPY, '-m', 'synth.prepare', str(path), '--submit'], 'campaign prepare/submit')
        if out is None:
            self.state['campaign_failed'] = True
        else:
            self.log(f'campaign {self.info["campaign_id"]} submitted ({path})')
        self.save()

    # ---- supply: stop the round once the stream covers the budget
    def accepted(self):
        from .roundpack import RoundFiles
        work = self.rd / 'work'
        packed = RoundFiles(self.rd).index(reload=True)['specs']   # packed specs: from the pack index (#245)
        for sid, (_, _, status, passed) in packed.items():
            self.records.setdefault(sid, status == 'valid' and passed)
        try:
            names = os.listdir(work)
        except OSError:
            names = []
        for sid in names:
            if sid in self.records:
                continue
            p = work / sid / 'record.json'
            if p.exists():
                r = read_json(p)
                if r is not None:
                    self.records[sid] = r.get('status') == 'valid' and bool((r.get('judge') or {}).get('pass'))
        return sum(self.records.values())

    def supply(self, progress):
        """Projected saved tokens of what the campaign has or will get: its own projection for opened tasks, plus
        streamed but unopened tasks and accepted but unpublished ones at the running tokens per trace."""
        pt = progress.get('per_task') or {}
        est = pt.get('tokens_per_trajectory_estimate') or self.info['tokens_per_trace_prior']
        m = self.cfg['recipe']['traces_per_task']
        opened, available = pt.get('tasks', 0), pt.get('available_tasks', 0)
        manifest = read_json(self.stream / 'manifest.json', {})
        rejected = read_json(self.stream / 'producer' / 'rejected.json', {})
        unpublished = max(0, self.accepted() - len(manifest) - len(rejected))
        return (pt.get('projected_tokens', 0) + max(0, available - opened) * m * est + unpublished * m * est,
                {'est': est, 'opened': opened, 'available': available, 'unpublished': unpublished})

    def stop_round(self, why):
        if (self.rd / 'STOP').exists():
            return
        (self.rd / 'STOP').write_text(why + '\n')
        self.state['round_stopped'] = {'time': time.time(), 'why': why}
        self.save()
        self.log(f'round STOP: {why}')

    def watch(self, progress):
        if progress:
            saved = progress.get('train_tokens_saved') or 0
            for mark in self.info['milestones']:
                key = str(mark)
                if saved >= mark and key not in self.state.setdefault('milestones', {}):
                    self.state['milestones'][key] = {'time': time.time(), 'train_tokens_saved': saved,
                                                     'trajectories': progress.get('train_trajectories')}
                    self.save()
                    self.log(f'MILESTONE {mark:,} saved tokens passed: {saved:,} saved, '
                             f'{progress.get("train_trajectories")} trajectories')
            if not (self.rd / 'STOP').exists():
                budget = self.info['tokens_budget']
                total, parts = self.supply(progress)
                if progress.get('train_stop'):
                    self.stop_round('the campaign closed training (budget reached or closed by hand)')
                elif total >= budget * (1 + self.cfg['stop']['supply_margin']):
                    self.stop_round(f'supply {total / 1e6:.0f}M >= budget {budget / 1e6:.0f}M x '
                                    f'{1 + self.cfg["stop"]["supply_margin"]:.2f} ({parts})')
        if not self.state.get('round_finished') and (self.rd / 'job.json').exists() and round_finished(self.rd):
            self.state['round_finished'] = time.time()
            self.save()
            self.log('round finished: every spec has its record')
            if progress and not (self.rd / 'STOP').exists():
                total, parts = self.supply(progress)
                if total < self.info['tokens_budget']:
                    self.log(f'ALERT the round ran out of specs before the budget: supply {total / 1e6:.0f}M < '
                             f'{self.info["tokens_budget"] / 1e6:.0f}M ({parts}); plan a top-up run')
        self.decontam_trim()
        if self.state.get('round_finished') and not self.decontam_hold():
            self.release_servers('the round has finished (task generation done; no switch to Super, #137)'
                                 if not self.state.get('decontam_kept') else 'the decontam hold is gone')
        if self.state.get('round_finished'):
            job = read_json(BINDS / self.info['bind'] / 'job.json')
            if job and (job.get('args') or {}).get('round') == self.info['round'] and alive(job['job_id']):
                subprocess.run(['condor_rm', job['job_id']], capture_output=True)
                self.log(f'bind job {job["job_id"]} removed (round finished)')
            self.remove_leftover_shards()

    def remove_leftover_shards(self):
        """Once the round has finished, its shards exit by themselves; one that is idle or held then (e.g. released
        after a memory hold by its periodic_release) would only wait up to 3 h for servers already released. Remove
        those, on every pass while there are any (only this round's cluster, from its job.json)."""
        job = read_json(self.rd / 'job.json')
        if not job:
            return
        cluster = str(job['job_id']).split('.')[0]
        st = job_states(cluster)
        if not st:
            return   # schedd unreachable, or the cluster is gone
        left = [f'{cluster}.{p}' for p, s in sorted(st.items()) if s in (1, 5)]
        done = []
        for j in left:
            try:
                r = subprocess.run(['condor_rm', j], capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                continue
            if r.returncode == 0:
                done.append(j)
        if done:
            self.log(f'round finished: removed {len(done)} idle or held round jobs ({", ".join(done)})')

    def quota(self):
        """ALERT when /data's free file quota runs low, at most hourly. A warning only (user, 2026-11-04): the run
        keeps going; freeing files is done by hand."""
        files = fast_files()
        if not files:
            return
        q, free = quota_cfg(self.cfg), files[1] - files[0]
        self.state['fast_files'] = {'time': time.time(), 'used': files[0], 'limit': files[1]}
        if free < q['alert_free_files'] and time.time() - self.state.get('quota_alerted', 0) > 3600:
            self.state['quota_alerted'] = time.time()
            self.log(f'ALERT /data file quota low: {free:,} of {files[1]:,} files free (< {q["alert_free_files"]:,})')
        self.save()

    def loop(self, every=300):
        self.log(f'driver up: run {self.name}, round {self.info["round"]}, pool {self.pool}, campaign '
                 f'{self.info["campaign_id"]}, budget {self.info["tokens_budget"]:,}, code {self.code}')
        passes = 0
        while True:
            passes += 1
            try:   # the driver's own heartbeat (production status, #237): its log is quiet between events
                write_json(self.run / 'heartbeat.json', {'time': time.time(), 'pass': passes, 'pid': os.getpid()})
            except OSError:
                pass
            try:
                if not self.state.get('round_finished'):
                    self.servers_up()
                    if self.servers_ready():
                        self.bind_up()
                        self.round_up()
                self.producer_up()
                self.campaign_up()
                progress = read_json(self.camp / 'ops' / 'progress.json')
                if not self.state.get('round_finished'):
                    self.quota()
                self.watch(progress)
                if progress and progress.get('phase') == 'complete':
                    if self.decontam_hold():   # the campaign is done before the decontam last pass: wait for it
                        if not self.state.get('complete'):
                            self.state['complete'] = time.time()
                            self.log('campaign complete; waiting for the decontam hold before releasing the servers')
                            self.save()
                        self.nap(every)
                        continue
                    self.state['complete'] = time.time()
                    self.release_servers('campaign complete')
                    self.save()
                    saved = progress.get('train_tokens_saved') or 0
                    if saved < self.info['tokens_budget']:
                        self.log(f'ALERT campaign complete below budget: {saved:,} < {self.info["tokens_budget"]:,}')
                    self.log(f'campaign complete: {saved:,} saved tokens, {progress.get("train_trajectories")} '
                             f'trajectories; driver done')
                    return
            except Exception as e:  # noqa: BLE001  (a transient Lustre or condor error must not end the driver)
                self.log(f'ALERT driver pass failed: {type(e).__name__}: {e}')
            self.nap(every)


def file_budget(info, files):
    """(files the run still needs, free files, message) at launch: the round's tasks not started yet x
    files_per_task, plus the campaign's loose files unless it is complete (a relaunch of a running run counts only
    what is left). `files`: fast_files()."""
    q = quota_cfg(info['config'])
    rd, camp = ROUNDS / info['round'], SYNTH / info['campaign_id']
    try:
        started = len(os.listdir(rd / 'work'))
    except OSError:
        started = 0
    from .roundpack import load_index
    started = max(started, len(load_index(rd)['specs']))
    complete = (read_json(camp / 'ops' / 'progress.json') or {}).get('phase') == 'complete'
    need = max(0, info['tasks_needed'] - started) * q['files_per_task'] + (0 if complete else q['campaign_loose_files'])
    free, tasks = files[1] - files[0], max(0, info['tasks_needed'] - started)
    parts = f'{tasks:,} tasks x {q["files_per_task"]}' + ('' if complete else f' + {q["campaign_loose_files"]:,} campaign')
    return need, free, f'/data files: {files[0]:,} of {files[1]:,} used, {free:,} free; this run needs ~{need:,} more ({parts})'


def launch(a):
    run = RUNS / a.name
    info = read_json(run / 'run.json')
    if not info:
        raise SystemExit(f'{run}/run.json missing: run `plan` first')
    root_env = pin_run_root(info)   # the driver job gets the run's registry root (condor jobs run with getenv = false)
    job = read_json(run / 'driver.json')
    if job and alive(job['job_id']):
        raise SystemExit(f'driver {job["job_id"]} is running')
    files = fast_files()
    if files is None:
        print('WARNING /data file quota unknown (lfs quota failed): file budget not checked', flush=True)
    else:
        need, free, msg = file_budget(info, files)
        print(msg, flush=True)
        if need > free:   # a warning only (user, 2026-11-05): files are freed by hand
            print(f'WARNING the run needs ~{need:,} files and /data has {free:,} free: pack finished rounds '
                  f'(task_synthesis.pipeline.roundpack) or archive campaigns (synth.archive) before it runs out',
                  flush=True)
    code = frozen_checkout()
    sub = run / 'driver.sub'
    sub.write_text(f'''universe = vanilla
executable = {HPY}
arguments = -u -m task_synthesis.pipeline.production drive --name {a.name}
initialdir = {code}
output = {run}/driver.$(Cluster).out
error = {run}/driver.$(Cluster).err
log = {run}/driver.condor.log
request_cpus = 2
request_memory = 8192
request_disk = 10G
+JobBatchName = "tasksynth-{a.name}-driver"
+MaxRuntime = 604800
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH={code} PYTHONNOUSERSITE=1{serving.condor_env(root_env)}"
requirements = {' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in BAD_CPU_NODES)}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    job_id = re.search(r'cluster (\d+)', r.stdout).group(1) + '.0'
    write_json(run / 'driver.json', {'job_id': job_id, 'code': str(code), 'time': time.time()})
    print(f'driver job {job_id} (code {code}); log {run}/driver.log')


def status(a):
    """Every component's job, condor state, heartbeat, counters and alerts (#237): task_synthesis.pipeline.live."""
    from . import live
    live.status(a)


def replan(a):
    """Re-plan the unstarted specs of a live run (#238): task_synthesis.pipeline.live.replan."""
    from . import live
    live.replan(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('plan', help='draw the specs, mark the binding share, write run.json (submits nothing)')
    p.add_argument('--name', required=True)
    p.add_argument('--tokens', type=int, required=True, help='Super saved-token budget of the campaign')
    p.add_argument('--milestones', type=int, nargs='*', help='saved-token marks whose crossing time is recorded')
    p.add_argument('--config', default=str(CONFIG))
    p.add_argument('--seed', type=int, help='spec seed (default: from the config seed and the run name)')
    p.add_argument('--specs', type=int, help='spec count (default: sized from the budget, with oversupply)')
    p.add_argument('--servers', help='existing generator servers as a pool (a+b) instead of starting the run\'s own; '
                                     'the driver then neither starts nor releases them')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('launch', help='submit the driver job')
    p.add_argument('--name', required=True)
    p.add_argument('--bid', type=int, default=35)
    p = sub.add_parser('drive', help='(inside the driver job)')
    p.add_argument('--name', required=True)
    p.add_argument('--every', type=int, default=300)
    p = sub.add_parser('status', help='every component: job, condor state, heartbeat, counters, alerts (#237)')
    p.add_argument('run', nargs='?')
    p.add_argument('--name')
    p.add_argument('--json', action='store_true', help='the full status as JSON (monitors read `alerts`)')
    p.add_argument('--no-metrics', action='store_true', help="don't read the generator servers' /metrics")
    p = sub.add_parser('replan', help='replace the specs nobody has started with new draws, live (#238)')
    p.add_argument('run', nargs='?')
    p.add_argument('--name')
    p.add_argument('--vocab', required=True, help='vocabulary of the new draws (specs.VOCAB_NAMES)')
    p.add_argument('--mix', help='minutes mix of the new draws (default: the run\'s recipe)')
    p.add_argument('--seed', type=int, required=True, help='seed of the new draws (unused by any round or re-plan)')
    p.add_argument('--asset-share', type=float, help='binding share of the new eligible draws (default: the recipe)')
    p.add_argument('--why', default='', help='reason, for run.json overrides_log')
    p.add_argument('--apply', action='store_true', help='do it (default: dry run, no writes)')
    p.add_argument('--restart-old-shards', action='store_true',
                   help='shards that cannot reload specs.jsonl (started before #238) are removed and their claims '
                        'taken over by a new round job (their in-flight sessions re-run from scratch)')
    p.add_argument('--confirm-minutes', type=float, default=5,
                   help='--apply: wait this long for every live shard to report the new specs.jsonl')
    a = ap.parse_args()
    if a.op in ('status', 'replan'):
        a.name = a.name or a.run
        if not a.name:
            ap.error(f'{a.op}: name the run')
    if a.op == 'plan':
        plan(a)
    elif a.op == 'launch':
        launch(a)
    elif a.op == 'drive':
        Driver(a.name).loop(a.every)
    elif a.op == 'replan':
        replan(a)
    else:
        status(a)


if __name__ == '__main__':
    sys.exit(main())
