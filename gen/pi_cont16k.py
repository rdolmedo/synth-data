#!/usr/bin/env python3
"""Run Harbor through a dedicated, lifetime-bound Pi continuation proxy.

Invoked automatically by AGENT_KIND=pi-cont16k gen/run_harbor.sh; same arguments.
Each rollout keeps its backend selected by the dispatcher, and its proxy logs are
copied with the job results. Existing agents and evaluation queues are unaffected.

Chain per rollout: Pi (sandbox) -> inference/sticky_proxy.py (per-turn cap, capture)
-> [tb_agents/response_budget_proxy.py (PI_RESPONSE_BUDGET)] -> tb_agents/pi_context_limit.py
(exact context guard; PI_CONTEXT_POLICY=compact: none) -> backend.

One launcher for evaluation and synthesis (#480; aa-evals' and task-synthesis' copies merged). Where the two
pipelines ran different defaults, PI_CONT16K_PROFILE picks them (each setting can still be set on its own):

  setting (environment)                         eval (default: aa_evals pi_eval,      synth (synth/worker.py sets
                                                eval_q35, M5 generation)              PI_CONT16K_PROFILE=synth)
  PI_PROXY_CAPTURE_FILES=0|1                    1: first-request dump, wire-responses/ 0 (PI_WIRE_LOG=1: wire.jsonl)
                                                (+ .meta.json), wire-requests.jsonl
  PI_PROXY_LOAD_POLLING=0|1                     1 (routing /metrics polls)             0 (--no-load-poll)
  PI_GATE_HEADERS=0|1                           1: X-Conversation-Id (one id per       0
                                                rollout) + X-Gate-Client (PI_GATE_CLIENT)
  PI_REQUEST_OVERRIDES_MODE=merge|update        merge (per-turn path, after the seed,  update (shallow update of
                                                chat_template_kwargs key by key)       every request)
  PI_PROXY_READY_TIMEOUT_SEC / _BUDGET_SEC      20 / 180                               2 / 60 (synth sets both)

Optional settings (environment; unset = off), both profiles:
  Lifecycle  PI_CONT16K_LOG_ROOT (log parent, default <jobs>/.pi-cont16k),
             PI_CONT16K_STOP_SIGNAL=TERM|INT, PI_CONT16K_STOP_TIMEOUT=10,
             PI_GRACEFUL_INTERRUPT=1 (Harbor child: SIGINT + 60 s before the normal stop),
             $PI_CONT16K_RUN/harbor-environment.json.
  Guard      PI_CONTEXT_POLICY=submit|compact, PI_CONTEXT_BACKEND_FILE (per-request endpoint file),
             PI_CONTEXT_EXACT_HEADROOM=1, PI_CONTEXT_RENDERER, PI_CONTEXT_TOKENIZER, PI_CONTEXT_STATE,
             PI_GUARD_BOUND_SHORTCUT=1, PI_FINISH_WARN_AT, PI_FINISH_RESERVE, PI_FINISH_STICKY (the #375 finish
             notice; see tb_agents/pi_context_limit.py), PI_RESPONSE_BUDGET (tb_agents/response_budget.py, #406).
  Proxy      PI_SAMPLING_SEED, PI_TEMPERATURE + PI_TOP_P (with the seed and no overrides; eval_q35 #167),
             PI_REASONING_EFFORT, PI_REQUEST_OVERRIDES (JSON), PI_CHAT_TEMPLATE_KWARGS (JSON
             merged into chat_template_kwargs), PI_MIRROR_REASONING=1, PI_INLINE_REASONING_HISTORY=1,
             PI_HTTP_FORCE_CLOSE=1, PI_DP_RANK (--dp-rank) or PI_CAMPAIGN_DP_RANK (X-data-parallel-rank header),
             PI_KV_GATE (HIGH,LOW) + PI_KV_GATE_FILE + PI_GATE_KEY (eval_q35 turn gate),
             PI_WIRE_LOG=1 (wire.jsonl: every served request + raw response in one file).
  Record     PI_MODEL_INPUT_JSON (aa_evals.common.pi_eval.model_input), PI_TASK_AGENT_CLASS_TASK.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tb_agents.pi_context_limit import (context_window, finish_policy, finish_settings, finish_sticky,
                                        start_context_guard)
from tb_agents.pi_continuation import VERSION, bundle_provenance, launch_cache, resampling_policy, verify_bundle
from aa_evals.common.pi_eval.model_input import from_env as model_input_from_env

PROFILES = ('eval', 'synth')


def profile(env=None):
    """PI_CONT16K_PROFILE: 'eval' (default) or 'synth'; it only sets the defaults the two pipelines differ on."""
    env = os.environ if env is None else env
    value = env.get('PI_CONT16K_PROFILE') or 'eval'
    if value not in PROFILES:
        raise SystemExit(f'PI_CONT16K_PROFILE must be one of {PROFILES}')
    return value


def _flag(env, key, default):
    value = env.get(key)
    if value in (None, ''):
        return default
    if value not in ('0', '1'):
        raise SystemExit(f'{key} must be 0 or 1')
    return value == '1'


def guard_policy(env=None):
    """The context guard's policy as protocol.json records it (#375 finish notice, as aa-evals' gen/pi_cont16k.py):
    context_policy and finish_notice ({warn_at, reserve[, sticky]} or None) from PI_FINISH_*."""
    warn_at, reserve = finish_settings(env)
    sticky = bool(warn_at or reserve) and finish_sticky(env)   # v4 sticky notices (finish-notices.jsonl = the map)
    return {'context_policy': finish_policy(warn_at, reserve, sticky),
            'finish_notice': {'warn_at': warn_at, 'reserve': reserve, **({'sticky': True} if sticky else {})}
            if (warn_at or reserve) else None}


def usage_summary(logs, target):
    """Harbor sees only the selected response; account for discarded samples separately."""
    log = logs / 'turns.jsonl'
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    discarded = sum(int(u.get('completion_tokens', u.get('output_tokens', 0)))
                    for r in records for u in r.get('sample_usage', [])[:-1])
    results = [json.loads(p.read_text()) for p in target.glob('*__*/result.json')]
    delivered = sum((r.get('agent_result') or {}).get('n_output_tokens') or 0 for r in results)
    return {'delivered_output_tokens': delivered if results else None,
            'discarded_output_tokens': discarded,
            'backend_output_tokens': delivered + discarded if results else None,
            'capped_requests': len(records),
            'note': 'Counts completed responses with reported usage; backend work aborted before usage arrives is not measurable here.'}


def stop_settings(env=None):
    env = os.environ if env is None else env
    name = env.get('PI_CONT16K_STOP_SIGNAL', 'TERM').upper().removeprefix('SIG')
    if name not in ('TERM', 'INT'):
        raise SystemExit('PI_CONT16K_STOP_SIGNAL must be TERM or INT')
    return getattr(signal, 'SIG' + name), float(env.get('PI_CONT16K_STOP_TIMEOUT', '10'))


def stop(process, graceful=False):
    if process is None or process.poll() is not None:
        return
    if graceful:
        # Let Harbor flush interrupted trials before the ordinary stop.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            pass
    sig, timeout = stop_settings()
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def wait_for_proxy(proxy, api, backend, timeout, budget=60, log=None):
    """Wait for a real backend response; busy inference can delay /models. `timeout` per request, `budget` in all."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + budget
    while True:
        if proxy.poll() is not None:
            raise RuntimeError('Continuation proxy exited during readiness check' + (f'; see {log}' if log else ''))
        try:
            with opener.open(api + '/models', timeout=timeout) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Continuation endpoint did not become ready: {backend}')
        time.sleep(.2)


def ready_timeout(env=None):
    """Seconds per /v1/models probe (PI_PROXY_READY_TIMEOUT_SEC). eval: 20 (a busy server can take seconds to answer
    /models; the DeepSeek campaign measured 2.4 s against a 2 s probe); synth: 2 (synth/worker.py sets its own)."""
    env = os.environ if env is None else env
    value = float(env.get('PI_PROXY_READY_TIMEOUT_SEC') or (20 if profile(env) == 'eval' else 2))
    if value <= 0:
        raise ValueError('PI_PROXY_READY_TIMEOUT_SEC must be positive')
    return value


def ready_budget(env=None):
    """Total seconds the proxy's first /models may take (PI_PROXY_READY_BUDGET_SEC). eval: 180; synth: 60. m1-hero
    (#244): 1,545 Super attempts (1.3%) ended here at 60 s with 2 s requests, on worker hosts whose /data client was
    saturated (synth/worker.py now sets 300)."""
    env = os.environ if env is None else env
    value = float(env.get('PI_PROXY_READY_BUDGET_SEC') or (180 if profile(env) == 'eval' else 60))
    if value <= 0:
        raise ValueError('PI_PROXY_READY_BUDGET_SEC must be positive')
    return value


def proxy_options(logs, env=None, conversation_id=None):
    """inference/sticky_proxy.py arguments beyond the per-turn cap, and their protocol record. Only flags that differ
    from the proxy's defaults are passed."""
    env = os.environ if env is None else env
    evaluation = profile(env) == 'eval'
    args, record = [], {'launcher_profile': profile(env)}
    capture = _flag(env, 'PI_PROXY_CAPTURE_FILES', evaluation)
    if capture:   # eval_q35/score.py and aa_evals pi_eval/score.py read wire-responses/
        args += ['--dump-dir', str(logs / 'first-request'), '--dump-max', '1',
                 '--response-dir', str(logs / 'wire-responses'),
                 '--request-events', str(logs / 'wire-requests.jsonl')]
    record['capture_files'] = capture
    polling = _flag(env, 'PI_PROXY_LOAD_POLLING', evaluation)
    if not polling:
        args += ['--no-load-poll']   # one guarded backend: routing polls cannot change anything
    record.update(routing_load_polling=polling, background_load_polling=polling,
                  single_backend_load_polling=polling)
    if _flag(env, 'PI_GATE_HEADERS', evaluation):
        # This proxy serves exactly one rollout. Identity survives identical model answers and transport retries.
        conversation_id = conversation_id or uuid.uuid4().hex
        args += ['--conversation-id', conversation_id, '--gate-client', env.get('PI_GATE_CLIENT', 'pi')]
        record['gate_conversation_id'] = conversation_id
    if env.get('PI_SAMPLING_SEED'):
        args += ['--sampling-seed', env['PI_SAMPLING_SEED']]
    if env.get('PI_REASONING_EFFORT'):   # eval_q35 k2_horizon: high (the proxy's default is max)
        args += ['--reasoning-effort', env['PI_REASONING_EFFORT']]
    sampling = [x for x in ('PI_TEMPERATURE', 'PI_TOP_P') if env.get(x)]
    if sampling and (not env.get('PI_SAMPLING_SEED') or env.get('PI_REQUEST_OVERRIDES')):
        # the proxy sets temperature / top_p only on the seeded path without registry overrides (eval_q35)
        raise SystemExit(f'{"/".join(sampling)} needs PI_SAMPLING_SEED and no PI_REQUEST_OVERRIDES')
    if env.get('PI_TEMPERATURE'):   # eval_q35 per-attempt temperature (#167; the proxy's default 1.0)
        args += ['--temperature', env['PI_TEMPERATURE']]
    if env.get('PI_TOP_P'):   # (the proxy's default 0.95)
        args += ['--top-p', env['PI_TOP_P']]
    if env.get('PI_SAMPLING_SEED') and not env.get('PI_REQUEST_OVERRIDES'):
        record['temperature'] = float(env.get('PI_TEMPERATURE') or 1.0)
        record['top_p'] = float(env.get('PI_TOP_P') or 0.95)
    overrides = json.loads(env.get('PI_REQUEST_OVERRIDES') or '{}')
    if env.get('PI_REQUEST_OVERRIDES'):   # the model's registry request settings (aa_evals M3.2, synth teachers)
        mode = env.get('PI_REQUEST_OVERRIDES_MODE') or ('merge' if evaluation else 'update')
        if mode not in ('merge', 'update'):
            raise SystemExit('PI_REQUEST_OVERRIDES_MODE must be merge or update')
        if mode == 'update' and env.get('PI_SAMPLING_SEED'):
            raise SystemExit('PI_REQUEST_OVERRIDES_MODE=update does not combine with PI_SAMPLING_SEED (the seed belongs '
                             'in the overrides; sticky_proxy refuses it)')
        args += ['--request-overrides', env['PI_REQUEST_OVERRIDES'],
                 *(['--request-overrides-mode', mode] if mode != 'merge' else [])]
        record['request_overrides_mode'] = mode
    record['request_overrides'] = overrides
    record['reasoning_effort'] = (overrides.get('reasoning_effort') if env.get('PI_REQUEST_OVERRIDES')
                                  else env.get('PI_REASONING_EFFORT') or 'max')
    if env.get('PI_CHAT_TEMPLATE_KWARGS'):
        args += ['--chat-template-kwargs', env['PI_CHAT_TEMPLATE_KWARGS']]
    if env.get('PI_MIRROR_REASONING') == '1':
        args += ['--mirror-reasoning-content']
    if env.get('PI_INLINE_REASONING_HISTORY') == '1':   # registry adapter (Nemotron Super)
        args += ['--inline-reasoning-history']
    record['history_reasoning_adapter'] = ('inline' if env.get('PI_INLINE_REASONING_HISTORY') == '1' else
                                           'mirror' if env.get('PI_MIRROR_REASONING') == '1' else None)
    if env.get('PI_DP_RANK'):   # eval_q35 snowball on vLLM data-parallel engines: one rank per rollout
        args += ['--dp-rank', env['PI_DP_RANK']]
    if env.get('PI_CAMPAIGN_DP_RANK') is not None:   # synth: the same pin as a backend header
        args += ['--backend-headers', json.dumps({'X-data-parallel-rank': env['PI_CAMPAIGN_DP_RANK']})]
        record['data_parallel_rank'] = int(env['PI_CAMPAIGN_DP_RANK'])
    if env.get('PI_HTTP_FORCE_CLOSE') == '1':   # as campaign 20261015T215325Z: no keep-alive reuse races
        args += ['--backend-force-close']
    record['backend_force_close'] = env.get('PI_HTTP_FORCE_CLOSE') == '1'
    if env.get('PI_KV_GATE'):   # HIGH,LOW: hold requests while the server's KV is full (eval_q35)
        args += ['--kv-gate', env['PI_KV_GATE']]
        if env.get('PI_KV_GATE_FILE'):   # KV reading published by the serving job (no /metrics polling)
            args += ['--kv-gate-file', env['PI_KV_GATE_FILE']]
        if env.get('PI_GATE_KEY'):   # eval_q35 context-aware turn gate (its address: kv.json gate_url)
            args += ['--gate-key', env['PI_GATE_KEY']]
    record['sampling_seed'] = env.get('PI_SAMPLING_SEED')
    record['kv_gate'] = env.get('PI_KV_GATE')
    if env.get('PI_WIRE_LOG') == '1':
        args += ['--wire-log', str(logs / 'wire.jsonl')]
    record['wire_log'] = env.get('PI_WIRE_LOG') == '1'
    return args, record


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 5:
        raise SystemExit('Expected tasks_dir jobs_dir job_name api_base concurrency [Harbor arguments]')
    tasks, jobs, name, backend, _ = args[:5]
    resamples, policy_source = resampling_policy()
    cap = int(os.environ.get('PI_CONT16K_CAP', '16384'))
    if not 1 <= cap <= 16384:
        raise SystemExit('PI_CONT16K_CAP must be 1..16384 (lower values are for smoke tests)')
    if os.environ.get('PI_VERSION', VERSION) != VERSION:
        raise SystemExit('pi-cont16k requires Pi 1.2.0')
    stop_settings()
    profile()
    proxy_options(Path(jobs), os.environ, conversation_id='check')   # bad settings fail before any log dir exists
    # PI_CONTEXT_POLICY: 'submit' (default) ends the episode at the context limit with
    # compaction disabled (tb_agents/pi_context_limit.py); 'compact' keeps Pi's normal
    # auto-compaction and only clamps max_tokens to the window in the proxy, the protocol of
    # campaign 20261015T215325Z (eval_q35/).
    policy = os.environ.get('PI_CONTEXT_POLICY', 'submit')
    if policy not in ('submit', 'compact'):
        raise SystemExit("PI_CONTEXT_POLICY must be 'submit' or 'compact'")
    backend_file = os.environ.get('PI_CONTEXT_BACKEND_FILE')
    if backend_file and (os.environ.get('PI_RESPONSE_BUDGET') or policy == 'compact'):
        # the guard would read the model endpoint itself, bypassing the response budget proxy / no guard runs
        raise SystemExit('PI_CONTEXT_BACKEND_FILE needs PI_CONTEXT_POLICY=submit and no PI_RESPONSE_BUDGET')
    cache = launch_cache()
    sha = verify_bundle(cache)
    # Fail before generating if the caller accidentally supplied the short-limit bundle.
    import tomllib
    for config in Path(tasks).glob('*/task.toml'):
        if tomllib.loads(config.read_text()).get('agent', {}).get('timeout_sec') != 21600:
            raise SystemExit(f'pi-cont16k requires the six-hour task bundle: {config}')
    parent = Path(os.environ.get('PI_CONT16K_LOG_ROOT') or Path(jobs) / '.pi-cont16k')
    parent.mkdir(parents=True, exist_ok=True)
    logs = Path(tempfile.mkdtemp(prefix=name + '-', dir=parent))
    listener = socket.socket()
    listener.bind(('0.0.0.0', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    host = socket.gethostbyname(socket.getfqdn())
    api = f'http://{host}:{port}/v1'
    env = {**os.environ, 'PI_CONT16K_PROXY_READY': '1', 'PI_CONT16K_LOCAL_API': api, 'PI_CONT16K_CAP': str(cap),
           'PI_CONT16K_RESAMPLES': str(resamples),
           'PI_CONT16K_CACHE': str(cache), 'MAX_TOKENS': str(cap), 'PI_VERSION': VERSION,
           'AGENT_KIND': 'pi-cont16k', 'PI_CONTEXT_POLICY': policy,
           'PI_MODEL_CLOCK_URL': f'http://127.0.0.1:{port}/proxy/model-clock'}
    # Opt-in per-run adapter. Already-running inbox workers pick this up on
    # their next rollout; campaigns without this file retain their environment.
    environment_config = None
    if os.environ.get('PI_CONT16K_RUN'):
        config_path = Path(os.environ['PI_CONT16K_RUN']) / 'harbor-environment.json'
        if config_path.exists():
            environment_config = json.loads(config_path.read_text())
            env['HARBOR_ENVIRONMENT'] = environment_config['class']
            # task-synthesis' per-run adapter decides the separator either way (the shared harness default is on)
            env['PI_CLI_PROMPT_SEPARATOR'] = '1' if environment_config.get('pi_cli_prompt_separator') else '0'
    timeout, budget = ready_timeout(), ready_budget()
    extra, proxy_record = proxy_options(logs)
    protocol = {'variant': 'pi-cont16k', 'pi_version': VERSION, **bundle_provenance(cache, sha),
                'max_output_tokens': cap, 'proxy_resamples': resamples, 'proxy_continuations': 0,
                'resampling_policy_source': policy_source,
                'agent_timeout_sec': 21600, 'backend': backend, 'proxy_api': api,
                'proxy_ready_timeout_sec': timeout, 'proxy_ready_budget_sec': budget, **proxy_record,
                'started_at': time.time(), 'smoke_cap': cap != 16384}
    modalities = model_input_from_env(env)
    if modalities is not None:
        protocol['model_input'] = modalities
    if env.get('PI_TASK_AGENT_CLASS_TASK'):
        protocol['task_agent_class'] = {'task': env['PI_TASK_AGENT_CLASS_TASK'],
                                        'class': env['PI_CONT16K_CLASS']}
    if environment_config is not None:
        protocol['harbor_environment'] = environment_config
    protocol['context_window'] = context_window()
    (logs / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    proxy = child = context_guard = response_guard = None

    def interrupted(signum, frame):
        raise InterruptedError(f'signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        if os.environ.get('PI_RESPONSE_BUDGET'):
            from tb_agents.response_budget_proxy import start as start_response_guard
            response_guard, backend = start_response_guard(backend, logs)
            protocol['response_budget'] = json.loads(os.environ['PI_RESPONSE_BUDGET'])
        window = context_window()
        if policy == 'compact':
            protocol.update(context_window=window, context_limit_action='compact',
                            compaction_enabled=True, context_policy='pi-normal-compaction')
            extra += ['--context-window', str(window)]
            guarded_backend = backend.rstrip('/').removesuffix('/v1')
        else:
            exact_output_cap = cap if os.environ.get('PI_CONTEXT_EXACT_HEADROOM') == '1' else None
            protocol.update(context_window=window, proxy_context_termination_cap=window,
                            context_limit_action='submit', compaction_enabled=False, **guard_policy(),
                            backend_file=backend_file, exact_headroom_output_cap=exact_output_cap,
                            backend_selection=('per-request endpoint file; one backend for tokenization and inference'
                                               if backend_file else 'fixed'))
            (logs / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
            context_guard, guarded_backend = start_context_guard(backend, logs, backend_file=backend_file,
                                                                output_cap=exact_output_cap,
                                                                state_file=os.environ.get('PI_CONTEXT_STATE'))
        (logs / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        with (logs / 'proxy.log').open('w') as output:
            proxy = subprocess.Popen([sys.executable, '-u', str(REPO / 'inference/sticky_proxy.py'),
                '--port', str(port), '--listen-fd', str(listener.fileno()),
                '--backends', guarded_backend,
                '--turn-max-tokens', str(cap), '--turn-retries', str(resamples), '--turn-continue', '0',
                '--turn-log', str(logs / 'turns.jsonl'), '--req-log', str(logs / 'requests.jsonl'),
                '--shutdown-timeout', '2', *extra], stdout=output, stderr=subprocess.STDOUT,
                pass_fds=(listener.fileno(),), start_new_session=True)
            listener.close()
            wait_for_proxy(proxy, f'http://127.0.0.1:{port}/v1', backend, timeout, budget, log=logs / 'proxy.log')
            args[3] = api
            child = subprocess.Popen(['bash', str(REPO / 'gen/run_harbor.sh'), *args], env=env,
                                     start_new_session=True)
            while child.poll() is None:
                if proxy.poll() is not None:
                    raise RuntimeError('Continuation proxy exited during the trial')
                time.sleep(.5)
            return child.returncode
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        stop(child, graceful=os.environ.get('PI_GRACEFUL_INTERRUPT') == '1')
        stop(proxy)
        stop(context_guard)
        stop(response_guard)
        listener.close()
        target = Path(jobs) / name
        if target.is_dir():
            (logs / 'usage.json').write_text(json.dumps(usage_summary(logs, target), indent=2) + '\n')
            shutil.move(str(logs), str(target / 'pi-cont16k'))


if __name__ == '__main__':
    sys.exit(main())
