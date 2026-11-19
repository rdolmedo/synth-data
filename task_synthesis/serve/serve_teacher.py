"""Standalone teacher server for task synthesis: one HTCondor GPU job running one vLLM from a synth teacher spec.

  python -m task_synthesis.serve.serve_teacher submit --teacher glm53-flash --name gen-glm
  python -m task_synthesis.serve.serve_teacher run --teacher glm53-flash --name gen-glm   (inside the job)
  python -m task_synthesis.serve.serve_teacher submit --teacher snowball-eval --checkpoint DIR --name NAME [--dry-run]
      (--checkpoint: one spec for every checkpoint of an architecture; the spec's [checkpoint_checks] must pass;
      omitted --gpus / --bid come from the spec's serving_resources.default_gpus / default_bid, else 4 / 63)
  python -m task_synthesis.serve.serve_teacher submit --pipeline task-synthesis --teacher deepseek-v41-flash-1017 ...
  python -m task_synthesis.serve.serve_teacher status|wait|release --name NAME   (tbshared.serving, #445 item 3):
      status: ready / starting / stale / stopped / unknown; wait [--timeout S]: exit 0 once ready (start dependents
      after it), 1 at the timeout or when the job left the queue; release --reason R: condor_rm the job.json job if
      it is ours and still queued (recorded in <home>/released.jsonl)

One launcher for every pipeline's model servers (#480, #445). --pipeline (default aa-serving: AA evals, M5, c30 and
every server since 11-14) picks the profile set and the submit defaults: a pipeline whose servers differ names its
own complete profile per teacher in PIPELINE_PROFILES (task-synthesis: DeepSeek one-shot task generation on vLLM
main, DSpark-5, no CPU KV tier, 20 CPUs / 320 GiB) and its defaults in PIPELINE_SUBMIT (task-synthesis: shapes filled to the binding
resource with tbshared condor_cost, no memory net, as on its branch). job.json records both, so a run that replaces
held servers knows whether condor releases a memory hold itself (memory_net; tbshared.holds.remove_held(has_net=)).

The job stages the checkpoint on node scratch, starts vLLM with the teacher's own command line
(synth.vllm_launch), probes it, and keeps <home>/endpoints.json fresh:
  {"served_name", "urls": [one per DP rank], "healthy_at", "updated_at", "job_id", "weights_model", "weights_revision",
   "weights_quantization", "numerics_sha256"}
The effective numerics of every engine (vLLM argv, the environment it set, weights and their quantization, template
sha256, vLLM build, GPU; synth/numerics.py) go to <home>/numerics.<job>.json and launch.json, with a stable
numerics_sha256 that endpoints.json repeats: a result that records it names the recipe that produced it, whatever
--tunables / --protocol the job was submitted with (job.json "overrides").
Clients treat a file older than 120 s as dead. Stop the server with condor_rm (receipt in <home>/job.json).
Submissions run from a frozen detached worktree, so edits to this checkout never reach a running job.

Resources (#378): every job logs its cgroup memory (rss, shmem, page cache, dirty) and CPU every 10 s to
<home>/resources.<job>.jsonl (tbshared/reslog.py; `python -I tbshared/reslog.py summary FILE` for peaks). Size a spec's
serving_resources from the non-reclaimable peak (`hard`), never from condor's RSS-only MemoryUsage. A job held for
memory is released with 25% more RAM, at most twice (tbshared.holds; MEMORY_* here are its values) when the
pipeline's submit has the memory net; --cpus / --memory-gib override the spec's shape. A --tunables (or
replica_tunables) kv_offloading_size moves the job's RAM by its CPU KV tier difference from the spec's, over every DP
rank (derived_memory_gib). A shape over 1.5x the derived one per GPU (CPUs or RAM) is refused with exit 2 unless
--allow-shape (#451 ws4); job.json "shape_check" records both shapes. Prompt-token sources (GPU prefix
cache, CPU KV tier) and generated tokens of every engine go to <home>/prompt-sources.<job>.jsonl every 5 minutes (#242).
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from synth.common import PROXY_ENV, PYTHON, atomic, read, swap_path
from synth.config import load_teacher
from synth.condor import requirements
from synth import engine_cache, numerics, venv_stage
from synth.engine_cache import fresh_vllm_root
from synth.gpu import assigned, expand, preflight
from synth.probe import probe, request
from synth.serve import consecutive_ports, dp_groups, gpu_name, stage_teacher, stop_process
from synth.vllm_launch import describe, replica_command, replica_env
from task_synthesis.common import ROOT as TS_ROOT, frozen_checkout
from tbshared import condor_cost, holds
from tbshared import serving as tbserving   # the registry (a local `serving` is the teacher's [serving] table)

TS_ROOT_CODE = Path(__file__).resolve().parents[2]   # the (frozen) checkout: aa_evals.qwenserve.gate runs from it

# $TASKSYNTH_SERVING_ROOT: another registry (dry runs, canaries); default the production one. Passed on to the job.
ROOT = Path(os.environ.get(tbserving.ROOT_ENV) or TS_ROOT / 'serving').absolute()   # absolute: the job runs elsewhere
PROTOCOL = {'context_tokens': 131072, 'max_output_tokens': 65536}
# Per-teacher serving profiles. DeepSeek runs its tuned production engine (the calibforge campaigns' tunables).
# max_output_tokens is a hard per-request cap in vLLM (--override-generation-config max_new_tokens bounds max_tokens):
# at the campaigns' 16,384, task replies with max reasoning were cut off (r1 aborted, 2026-10-30). Task synthesis
# sets no output cap of its own (user, 2026-10-30), so the cap is the whole context. Both DeepSeek profiles serve vLLM
# main (aa-serving since 2026-11-18, task synthesis since 2026-11-19, #478); each command publishes its own
# engine-cache snapshot at its first healthy start (no seed_from).
PROFILES = {
    # AA evals (#51): Qwen3.8-27B under test at its native 262k context (AA: the model's full window).
    'qwen38-27b-eval': {
        'protocol': {'context_tokens': 262144, 'max_output_tokens': 262144},
    },
    # AA evals (#70): the tuned recipe (synth/teachers/qwen38-27b-eval-fast.toml) behind the admission gate
    # (aa_evals/qwenserve/gate.py): clients use the gate's URL and a high in-flight cap of their own.
    'qwen38-27b-eval-fast': {
        'protocol': {'context_tokens': 262144, 'max_output_tokens': 262144},
        # #110: per-client KV share on (HLE +14% tok/s, queue p95 597 -> 381 s, no throughput loss).
        'gate': {'policy': {'client_share': True}},
    },
    # AA evals (#111): the same recipe on 1xH100 (synth/teachers/qwen38-27b-eval-h100.toml), gate in front with an
    # H100 policy: its 1.1M-token GPU KV fills with ~20 agents, so the kv_fit target starts at 1.2 (idle contexts
    # spill to the 80 GiB CPU tier; #70's 0.6 start is B200-sized) and the in-flight brakes sit at 0.98 / 0.94 with a
    # 0.97 decode budget. Replays: aa_evals/qwenserve/README.md (H100).
    'qwen38-27b-eval-h100': {
        'protocol': {'context_tokens': 262144, 'max_output_tokens': 262144},
        'gate': {'policy': {'kv_target_fraction_offload': 1.2, 'kv_target_min_offload': 0.8,
                            'emergency_kv_high': 0.98, 'resume_kv': 0.94, 'inflight_kv_max': 0.97}},
    },
    # AA evals (M3.3, #176): Nemotron 3.5 Lightning (base) on eval_q35's tuned v032 serving (synth/teachers/
    # nemotron35-lightning-eval.toml), 128k cap. v032's turn gate admitted contexts up to 1.5x the GPU cache (the CPU
    # tier holds the rest; serve_lightning/README.md: 1.5 led 0.7 by 6 min to 95% of a TB 2.1 replay): here the gate's
    # kv_fit target starts at 1.5 with a floor of 1.0, the rest of the gate policy at its defaults.
    'nemotron35-lightning-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'gate': {'policy': {'kv_target_fraction_offload': 1.5, 'kv_target_min_offload': 1.0}},
    },
    # AA evals M3.3 (#176): Qwen3.6-35B-A3B at the M3.0 128k context, gate v2.1 (per-client KV share) with #111's H100
    # policy: on a mixed replay (64 agents + 32 HLE + 8 short, H100 NVL) the default policy held KV at 0.68 and lost 12%
    # (1,635 vs 1,850 gen tok/s without a gate, 17 HLE 503s); the H100 policy 1,854, KV 0.96, no preemptions, no 503s.
    'qwen36-35b-a3b-eval-h100': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'gate': {'policy': {'client_share': True, 'kv_target_fraction_offload': 1.2, 'kv_target_min_offload': 0.8,
                            'emergency_kv_high': 0.98, 'resume_kv': 0.94, 'inflight_kv_max': 0.97}},
    },
    # AA evals M3.3 (#176): Qwen3.6-35B-A3B on 1xB200 (qwen36-35b-a3b-eval.toml): #70's B200 gate policy.
    'qwen36-35b-a3b-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'gate': {'policy': {'client_share': True}},
    },
    # AA evals M3.3 (#176): Qwen3.5-4B, eval_q35's fp8kv serving at 128k. No gate here: its admission is eval_q35's
    # context-aware turn gate, which pi_eval runs with the eval (registry request.ctx_gate).
    'qwen35-4b-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
    },
    # AA evals (M3.3, #176): Gemma 4 26B-A4B-it under test at the 128k eval cap (synth/teachers/
    # gemma4-26b-a4b-it-eval.toml), the default gate in front (on H100 it held 1.5-1.8k gen tok/s from 64 to 128 agents
    # where no gate fell to 1.3k; a fitted target won at 64 and lost past it). Tuning: aa_evals/gemmaserve/README.md.
    'gemma4-26b-a4b-it-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'gate': {'policy': {}},
    },
    # Its 1xB200 variant (FP8 + FP8 KV + MTP2): the same gate (the replay could not test admission on B200: README).
    'gemma4-26b-a4b-it-eval-b200': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'gate': {'policy': {}},
    },
    # AA evals (#33): the same recipe with a bf16 KV cache, testing the EOS-inside-reasoning defect.
    'qwen38-27b-eval-fast-bf16kv': {
        'protocol': {'context_tokens': 262144, 'max_output_tokens': 262144},
        'gate': {'policy': {}},
    },
    # stage_venv (synth/venv_stage.py, #554) on the vLLM-main specs' profiles: a node-local copy of the venv, ~2 min less
    # start-up than imports from /data (A/B 145 / 149 vs 288 s, job 17980343); same files, same numerics. On
    # nemotron35-super-mm(-text), snowball-eval-main and deepseek-v41-flash-1017 (staged starts verified: Super 819 s
    # cold warm-up, Snowball 212 s smoke, DeepSeek task-synthesis 921 s smoke job 17980755: 4 DP ranks, launch -> engine
    # init 34 s vs ~160 s from /data).
    # AA evals (#34 GDP.pdf): Super VL at its native 1M context, for long multi-page image prompts.
    'nemotron35-super-mm': {
        'protocol': {'context_tokens': 1048576, 'max_output_tokens': 65536},
        'tunables': {'stage_venv': 1},
        # #478: vLLM main's first start JIT-compiles FlashInfer modules and NVFP4 CuTe-DSL GEMMs (~11 min); a recipe
        # without its own snapshot yet (-mm, -mm-text, or either after an env change such as #557's eot) seeds from
        # 15282107916d460058f1: a staged -mm-text start (job 17980718) that holds both, built under the staged venv
        # path these profiles run (stage_venv). Was cbe361c0, an unstaged -mm start: its FlashInfer JIT modules carry
        # /data venv paths, so a staged engine rebuilt them (~5 min).
        'seed_from': '15282107916d460058f1',
    },
    'deepseek-v41-flash-1017': {
        'protocol': {'context_tokens': 132000, 'max_output_tokens': 132000},
        'tunables': {'prompt_tokens_details': 1, 'kv_offloading_size': 64,
                     'extra_env': {'VLLM_USE_SIMPLE_KV_OFFLOAD': '1'}, 'stage_venv': 1},
        # No seed_from (2026-11-18, #478): 8162cffd... is a vllm032 engine-cache snapshot and the spec now serves vLLM
        # main; the first healthy engine publishes main's own snapshot.
    },
    # AAI-open eval: GLM-5.3-Flash on the 2026-10-31 study's recipe (synth/teachers/glm53-flash-eval.toml), 128k context.
    # A 4xB200 job runs two DP2/EP2 engines. No front gate, like DeepSeek: clients pin rollouts per rank, and inventory
    # is held near the study's knee (~64 running per rank).
    'glm53-flash-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'tunables': {'prompt_tokens_details': 1},
    },
    # AA evals: Snowball-architecture SFT checkpoints (GrugMoe 67B-A2B, synth/teachers/snowball-eval.toml, one spec for
    # every checkpoint: submit --checkpoint) on eval_q35's Snowball TB recipe at the 128k eval cap. No gate here: like
    # qwen35-4b-eval, admission is eval_q35's context-aware turn gate, which pi_eval runs with the eval (request.ctx_gate).
    'snowball-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
    },
    # snowball-eval on vLLM main 8f3c672dc (synth/teachers/snowball-eval-main.toml, #478): same profile, B200 only.
    'snowball-eval-main': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
        'tunables': {'stage_venv': 1},
    },
    # Marin's post-trained Snowball releases (synth/teachers/datakit-eval.toml): snowball-eval with the checkpoint's
    # Datakit chat template, datakit parsers and thinking on.
    'datakit-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
    },
    # The Snowflakes (tiny GrugMoe, synth/teachers/snowflake-eval.toml): snowball-eval in bf16 weights and KV.
    'snowflake-eval': {
        'protocol': {'context_tokens': 131072, 'max_output_tokens': 131072},
    },
}
# Text-only eval variants (#348): synth/teachers/<slug>-text.toml is <slug> with --language-model-only (no vision
# tower, no image inputs) and otherwise the same recipe, so it runs under the same profile.
TEXT_ONLY_VARIANTS = ('qwen38-27b-eval-fast', 'nemotron35-super-mm', 'qwen36-35b-a3b-eval', 'gemma4-26b-a4b-it-eval-b200',
                      'qwen35-4b-eval', 'glm53-flash-eval')
PROFILES.update({f'{slug}-text': PROFILES[slug] for slug in TEXT_ONLY_VARIANTS})
# Per-pipeline profile sets (--pipeline): a complete profile per teacher, used instead of its PROFILES entry; a
# teacher a pipeline does not name runs its PROFILES entry. tbshared/numerics.py renders each pipeline's specs with
# its set (numerics_record.json sections aa-serving / task-synthesis).
PIPELINE_PROFILES = {
    'task-synthesis': {
        # Task synthesis (M1 production, task_synthesis/pipeline/production.py): one-shot task generation. The
        # protocol: max_output_tokens is a hard per-request cap in vLLM (--override-generation-config max_new_tokens
        # bounds max_tokens): at the campaigns' 16,384, task replies with max reasoning were cut off (r1 aborted,
        # 2026-10-30). Task synthesis sets no output cap of its own (user, 2026-10-30), so the cap is the whole
        # context.
        # Until 2026-11-19 (vllm032), spec_tokens 3 (issue #163): DSpark with 3 draft tokens instead of the campaigns' 5,
        # superseded on vLLM main (below: 5 ties 3, the profile uses the spec's 5). One-shot task generation
        # runs ~500 long single generations per DP rank; in a replay of 3,000 m1-hero requests at 2,000 running per
        # server (campaign 20261104-deepseek-v41-flash-1017-oneshot-spec163) it generated +6.5% tokens per second over
        # 5 (two runs, 0.1-0.6% apart), while 2 and no speculation were 1-4% below 5. Only up to ~512 running per
        # rank: at 2,500 per server (625 per rank) 3 was 3% below 5, because 4 tokens x 625 requests pads to the 3,072
        # CUDA graph (or trims drafts) while 5 fits 4,092. With DSpark-5 on main the 2,500 point is unmeasured (not
        # known to be worse): re-measure before raising one-shot admission above ~2,000 per server.
        'deepseek-v41-flash-1017': {
            'protocol': {'context_tokens': 132000, 'max_output_tokens': 132000},
            # No CPU KV tier (#242, 2026-11-06): one-shot task generation reuses little (prefix hit ~0.2-0.3, GPU KV <=
            # 24% at 2,000 running), so the 4 x 64 GiB tier served 0.0% of prompt tokens. Replay of 3,000 recorded
            # m1-hero requests at 2,000 running on one 4xB200 node (campaign
            # 20261105-deepseek-v41-flash-1017-oneshot-tier242): tier + 32 CPUs 72.8k / 71.8k gen tok/s, no tier + 32
            # CPUs 73.6k / 71.4k, no tier + 20 CPUs 73.0k / 71.6k.
            # vLLM main and DSpark-5 since 2026-11-19 (#478; spec runtime.venv and spec_tokens, as aa-serving): the same
            # replay (3,000 m1-hero requests at 2,000 running, one 4xB200 node cpu302, back to back; campaign
            # 20261118-deepseek-v41-flash-1017-oneshot-main478) gave vllm032 DSpark-3 73.4k / 71.3k gen tok/s, main
            # DSpark-3 77.5k / 79.3k, main DSpark-5 78.9k / 78.8k: main +8.4% (k3) / +9.0% (k5) on the means (+5.6% to
            # +11.3% per window); 5 ties 3 (+0.6%, inside the windows'
            # spread; equal latency p50/p90; 3 has lower ITL 60-66 vs 72-74 ms, 5 longer accepted runs 2.8-2.9 vs
            # 2.3-2.6). Main no longer charges draft slots to the token budget (vllm#59468), which is what made 3 win on
            # vllm032 (#163), so the override goes. Host memory on main: hard peak 223 GiB (vllm032 221), cores mean 5.3,
            # peak 12.3: the shape below holds. The replay's main arms copied the FlashInfer cubins to node-local disk
            # (a wrapper, ~265 s per start); production reads them from the venv with FLASHINFER_NO_DOWNLOAD (spec
            # header), so its first start is the smoke test of that path. Was venv vllm032 + spec_tokens 3 (2026-11-18).
            'tunables': {'prompt_tokens_details': 1, 'stage_venv': 1},   # stage_venv: as the aa-serving profile
            # Host memory without the tier (same replay, cgroup per minute): Engram tables 191 GiB in shared memory +
            # ~28 GiB of processes = ~219 GiB that cannot be reclaimed; the staged checkpoint's page cache (up to ~670
            # GiB) is reclaimable and fills any limit. 320 GiB / 20 CPUs (3.3-4.5 cores used, peak 6.4) = 20 units: a
            # server is 116 units/h instead of 144 at 768 GiB (-19%) and 192 on m1-hero's ds1 (1.5 TiB).
            'memory_gib': 320,
            # /dev/shm 200 GiB (+DevShmSize; the cluster's job wrapper sizes it to half the slot's memory otherwise, 160
            # GiB here): vLLM main shares its two offloaded Engram tables (layers 1 and 14, 94.42 GiB each) across the 4
            # DP ranks through /dev/shm only when their sum (193,365 MiB) fits (engram.py). At 160 GiB both main smokes
            # (17976930, 17980755) split them across the ranks instead (2 x 23.6 GiB per rank), a layout no replay
            # measured; the #478 replay (512 GiB job, 256 GiB /dev/shm) shared them, hard peak 223 GiB, which 320 holds.
            # The tables count in the job's memory either way, so this costs nothing (2026-11-19, #559 review).
            'dev_shm_gib': 200,
            'cpus': 20,
            # no seed_from (2026-11-19): 8162cffd... was a vllm032 engine-cache snapshot; main publishes its own
        },
    },
}
PIPELINES = ('aa-serving', 'task-synthesis')
DEFAULT_PIPELINE = 'aa-serving'
# Submit defaults per pipeline (each as on its branch; #480 "Defaults that differ"): fill = request CPUs and RAM at
# the binding resource within the GPU class's per-GPU share (tbshared condor_cost.fill_plan, tbshared 1.5); memory_net
# = condor releases a memory hold with more RAM (tbshared.holds). --fill / --no-fill override the first.
PIPELINE_SUBMIT = {
    'aa-serving': {'fill': False, 'memory_net': True},
    'task-synthesis': {'fill': True, 'memory_net': False},
}
def bad_nodes() -> list:
    """The nodes every server job excludes, read at submit time (#531 A9: a long-lived process keeps what it imported):
    tbshared's condor_cost.bad_nodes() (its file's BAD_NODES + the live file), the one list (cpu305, this launcher's own
    extra until 2026-11-19, is in it)."""
    return sorted(condor_cost.bad_nodes())


def profile_for(slug, pipeline=DEFAULT_PIPELINE):
    """The serving profile of `slug` for `pipeline`: its PIPELINE_PROFILES entry, else its PROFILES entry, else {}."""
    if pipeline not in PIPELINES:
        raise SystemExit(f'unknown pipeline {pipeline!r} (known: {", ".join(PIPELINES)})')
    own = PIPELINE_PROFILES.get(pipeline, {})
    return own[slug] if slug in own else PROFILES.get(slug, {})


def log(home, msg):
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(home / 'serve.log', 'a') as f:
        f.write(line + '\n')


def replica_groups(gpus, serving, tunables):
    """replicas mode: the GPU group of every engine. Tunable engines_per_gpu (k, default 1) starts k engines on each
    group, each with gpu_util / k (engine_tunables). For tiny models (the Snowflakes, 2026-11-16) one engine is bound
    by its single-threaded engine core (about 12 tok/s per request at 250 running while the GPU idles), so k engines
    on one GPU multiply the throughput per GPU."""
    per = serving.get('gpus_per_replica', 1)
    k = max(1, int(tunables.get('engines_per_gpu', 1)))
    return [gpus[i * per:(i + 1) * per] for i in range(len(gpus) // per) for _ in range(k)], k


def engine_tunables(tunables, k):
    """An engine's tunables when k engines share its GPU(s): the memory share split k ways. A spec without gpu_util
    (vLLM 0.34 sizes its KV after warmup, snowball-eval-main) runs one engine per GPU."""
    if k > 1 and 'gpu_util' not in tunables:
        raise ValueError(f'engines_per_gpu={k} needs a gpu_util tunable to split the GPU memory; this spec has none '
                         '(its engine sizes its KV cache itself): one engine per GPU')
    return {**tunables, 'gpu_util': round(float(tunables['gpu_util']) / k, 3)} if k > 1 else tunables


def wait_up(port, timeout):
    """True once the engine on `port` answers /v1/models (it has profiled memory and allocated its KV cache)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            request(f'http://127.0.0.1:{port}/v1/models', timeout=3)
            return True
        except OSError:
            time.sleep(5)
    return False


def dp_external_plan(gpus, serving, ports=consecutive_ports):
    """The DP engines of a dp_external_lb job, as (gpus, base API port, rank-port template values): every engine has
    its own API ports (one per rank), DP RPC port and supervisor port. `dp_group_gpus` splits the job's GPUs into
    engines (GLM-5.3-Flash: two DP2/EP2 engines on 4 GPUs); without it the job is one engine, as before."""
    taken = set()

    def block(n):
        while True:
            base = ports(n)
            span = set(range(base, base + n))
            if not span & taken:
                taken.update(span)
                return base

    per = serving.get('gpus_per_replica', 1)
    plan = []
    for group in dp_groups(gpus, serving):
        n = len(group) // per
        plan.append((group, block(n), {'dp_size': n, 'rpc_port': block(1), 'supervisor_port': block(1)}))
    return plan


def engine_plan(slug, teacher, gpus, *, pipeline=DEFAULT_PIPELINE, tunables_json='', protocol_json='',
                ports=consecutive_ports):
    """Every engine of a job on `gpus` (GPU ids) for `pipeline`: the teacher's serving, the pipeline's profile and the
    submit-time --tunables / --protocol JSON on top -> {'profile', 'protocol', 'tunables', 'per_replica', 'k',
    'groups', 'bases', 'extras', 'ports'}: engine i runs on groups[i] with API port bases[i] and rank_ports extras[i];
    `ports` lists every API port (one per DP rank)."""
    profile = profile_for(slug, pipeline)
    protocol = {**profile.get('protocol', PROTOCOL), **json.loads(protocol_json or '{}')}
    serving = teacher['serving']
    tunables = {**serving.get('tunables', {}), **profile.get('tunables', {}), **json.loads(tunables_json or '{}')}
    per = serving.get('gpus_per_replica', 1)
    n = len(gpus) // per
    k = 1
    if serving.get('mode') == 'dp_external_lb':
        plan = dp_external_plan(gpus, serving, ports)
        groups = [group for group, _, _ in plan]
        bases = [base for _, base, _ in plan]
        extras = [extra for _, _, extra in plan]
        all_ports = [base + i for _, base, extra in plan for i in range(extra['dp_size'])]
    elif serving.get('mode') == 'dp_internal':
        base = ports(1)
        extras, groups, bases, all_ports = [{'dp_size': n}], [gpus], [base], [base]
    else:
        # One vLLM process per replica (gpus_per_replica GPUs each). Before 2026-11-02 a multi-GPU replicas job
        # started a single TP1 process with every GPU visible, which served on the first GPU only.
        groups, k = replica_groups(gpus, serving, tunables)
        if k > 1:
            tunables = engine_tunables(tunables, k)
        extras = [{} for _ in groups]
        bases = all_ports = [ports(1) for _ in groups]
    return {'profile': profile, 'protocol': protocol, 'tunables': tunables,
            # tunable replica_tunables (#110): a list of per-replica overrides, e.g. [{}, {"mtp": 4}]
            'per_replica': tunables.get('replica_tunables') or [], 'k': k,
            'groups': groups, 'bases': bases, 'extras': extras, 'ports': all_ports}


def engine_command(slug, teacher, model_dir, gpus, pipeline=DEFAULT_PIPELINE, staged=None):
    """The vLLM command of the job's first engine on `gpus` GPUs (a count or a list of ids) for `pipeline`: the
    teacher's own command with the pipeline's profile on top -> (argv, tunables, ports). Task synthesis #182."""
    ids = [str(i) for i in range(gpus)] if isinstance(gpus, int) else list(gpus)
    plan = engine_plan(slug, teacher, ids, pipeline=pipeline)
    tun = {**plan['tunables'], **(plan['per_replica'][0] if plan['per_replica'] else {})}
    argv = replica_command(teacher, plan['protocol'], port=plan['bases'][0], model_dir=str(model_dir), tunables=tun,
                           rank_ports={**plan['extras'][0], **(staged or {})})
    return argv, tun, plan['ports']


def checkpoint_overrides(a):
    """load_teacher overrides of a per-checkpoint launch (--checkpoint): every {checkpoint} in the spec follows it."""
    return {'checkpoint': str(Path(a.checkpoint).resolve())} if a.checkpoint else None


def check_checkpoint(teacher):
    """The spec's [checkpoint_checks] on its (per-launch) checkpoint: config.json architectures and the sha256 of
    chat_template.jinja, which the spec's parsers read. Raises SystemExit naming what does not match."""
    checks = teacher.get('checkpoint_checks') or {}
    ckpt = Path(teacher['checkpoint'])
    problems = []
    if checks.get('architectures'):
        try:
            archs = json.loads((ckpt / 'config.json').read_text()).get('architectures') or []
        except (OSError, ValueError) as e:
            archs, problems = [], problems + [f'config.json unreadable: {e!r}']
        if not set(archs) & set(checks['architectures']):
            problems.append(f'architectures {archs} not in {checks["architectures"]}')
    if checks.get('chat_template_sha256'):
        import hashlib
        try:
            digest = hashlib.sha256((ckpt / 'chat_template.jinja').read_bytes()).hexdigest()
        except OSError as e:
            digest = f'unreadable ({e!r})'
        if digest not in checks['chat_template_sha256']:
            problems.append(f'chat_template.jinja sha256 {digest} not in {checks["chat_template_sha256"]}')
    if checks and not (ckpt / 'model.safetensors.index.json').exists() and not list(ckpt.glob('*.safetensors')):
        problems.append('no safetensors weights')
    if problems:
        raise SystemExit(f'{ckpt}: not a checkpoint for teacher {teacher["slug"]}: ' + '; '.join(problems))
    return {'checkpoint': str(ckpt), **({'checks': sorted(checks)} if checks else {})}


def served_identity(teacher, replicas):
    """The fields a client copies into its results to tell recipes apart: one served_name covers several (Qwen3.8-27B
    NVFP4 synth / FP8 eval, Gemma-4-26B BF16 judge / FP8 under test, Lightning FP8 on H100 / BF16 on B200)."""
    quant = sorted({(r.get('weights') or {}).get('quantization', {}).get('label') for r in replicas} - {None})
    return {'weights_model': teacher.get('model'), 'weights_revision': teacher.get('revision'),
            'weights_quantization': quant[0] if len(quant) == 1 else (quant or None),
            'numerics_sha256': numerics.job_numerics_sha256(replicas)}


RESLOG = TS_ROOT_CODE / 'tbshared' / 'reslog.py'   # the one sampler (#445 item 2); in the frozen checkout
RESLOG_SEC = 10


def start_sampler(home):
    """The cgroup sampler (tbshared/reslog.py) as a child of this process: memory (incl. page cache, shmem, dirty), CPU
    (cores, run-queue wait) and per-process RSS (--procs, ~every minute) of
    the whole job every RESLOG_SEC to <home>/resources.<job>.jsonl. Best effort: a sampler that fails to start never
    stops the server."""
    try:
        job = os.environ.get('SERVE_JOB_ID') or f'pid{os.getpid()}'
        return subprocess.Popen([sys.executable, '-I', str(RESLOG), 'run', str(home / f'resources.{job}.jsonl'),
                                 str(RESLOG_SEC), '--parent', str(os.getpid()), '--procs'], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return None


def stop_sampler(p):
    """Stop the sampler before this process exits: it must never outlive the job's main process (#401: an orphaned
    sampler kept a finished pi_eval worker's cgroup alive and condor held it). It also exits on its own within 1 s
    of losing its parent."""
    if p is None:
        return
    p.terminate()
    try:
        p.wait(5)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:   # stuck in a /data write: never hold up the server's exit for it
            pass


def run(a):
    home = ROOT / a.name
    home.mkdir(parents=True, exist_ok=True)
    sampler = start_sampler(home)
    try:
        serve(a, home)
    finally:
        stop_sampler(sampler)


def serve(a, home):
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.append(1))
    teacher = load_teacher(a.teacher, checkpoint_overrides(a))
    check_checkpoint(teacher)
    gpus = expand(assigned())
    passed, report = preflight(gpus)
    atomic(home / 'preflight.json', {**report, 'passed': passed, 'host': socket.getfqdn()})
    if not passed:
        log(home, f'GPU preflight failed: {report}')
        sys.exit(3)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR', f'/tmp/{os.environ.get("USER")}/serve-{a.name}'))
    local = engine_cache.stable_root(scratch)  # /tmp/synth when /tmp is per job: identical paths for cache snapshots
    pipeline = getattr(a, 'pipeline', None) or DEFAULT_PIPELINE
    t0 = time.time()
    # the checkpoint, stage_extra dirs ({stage_<name>} in the command: nemotron35-bolt's draft) and kernel wheels
    model_dir, staged, kernel_provider = stage_teacher(teacher, local)
    log(home, f'staged {teacher["checkpoint"]} in {time.time() - t0:.0f}s'
              + (f' (+ {", ".join(sorted(k for k in staged if k != "stage_model"))})' if len(staged) > 1 else ''))
    serving = teacher['serving']
    plan = engine_plan(a.teacher, teacher, gpus, pipeline=pipeline, tunables_json=a.tunables, protocol_json=a.protocol)
    profile, protocol, tunables, per_replica = plan['profile'], plan['protocol'], plan['tunables'], plan['per_replica']
    groups, bases, extras, ports = plan['groups'], plan['bases'], plan['extras'], plan['ports']
    if plan['k'] > 1:
        log(home, f'engines_per_gpu {plan["k"]}: {len(groups)} engines, gpu_util {tunables["gpu_util"]} each')
    gate = profile.get('gate')
    # tunable stage_venv: each engine's venv copied to the node first (synth/venv_stage.py: ~2 min less start-up for
    # vLLM main from /data); a venv that cannot be staged (no tar, tar and venv differ) runs from its source, logged
    staged_venvs = {}
    for v in venvs_to_stage(plan, teacher):
        t1 = time.time()
        path, rec = venv_stage.stage(v, local)
        staged_venvs[v] = (path, rec)
        log(home, f'venv {v} staged to {path} in {time.time() - t1:.0f}s' if path else
                  f'venv {v} not staged (runs from it): {rec.get("reason")}')
    procs, gates, launches = [], [], []
    # replica_tunables (#110): twin engines in one job serve different recipes, so an A/B runs on the same node, at
    # the same time, on identical load
    for i, (group, port) in enumerate(zip(groups, bases)):
        if i and groups[i] == groups[i - 1] and not wait_up(bases[i - 1], serving.get('startup_timeout_sec', 5400)):
            log(home, f'engine r{i - 1} did not come up: starting r{i} anyway')   # engines sharing a GPU start one
            # at a time, so each memory profile sees the reservations of the ones before it
        tun = {**tunables, **(per_replica[i] if i < len(per_replica) else {})}
        engine = local / 'engines' / f'r{i}'
        argv, env, source, copy, unstage = replica_launch(
            teacher, tun, protocol, gpus=group, port=port, model_dir=str(model_dir), rank_ports={**extras[i], **staged},
            scratch=engine, kernel_provider=kernel_provider, staged_venvs=staged_venvs)
        venv_rec = staged_venvs.get(source, (None, None))[1] if tun.get('stage_venv') else None
        key, facts = engine_cache.config_key(argv, env, str(copy or source), gpu_name(report, group))
        # its own snapshot, else (a first staged start) the /fast-venv recipe's (kernels keyed by path rebuild, the rest
        # is reused), else the profile's seed_from; every cache validates its own entries
        src_key = (engine_cache.config_key([unstage(a) for a in argv], {k: unstage(v) for k, v in env.items()},
                                           source, gpu_name(report, group))[0] if copy else None)
        seeded = engine_cache.seed_first(engine / 'cache', key, [src_key, profile.get('seed_from')])
        fresh_vllm_root(env, engine)
        vlog = scratch / f'vllm-r{i}.log'
        procs.append(subprocess.Popen(argv, env=env, stdout=vlog.open('ab'), stderr=subprocess.STDOUT,
                                      start_new_session=True))
        # Effective numerics (synth/numerics.py): what this engine runs after every override, with a stable hash
        rec = numerics.safe_replica_numerics(
            teacher, [unstage(a) for a in argv], {k: unstage(v) for k, v in env.items()}, model_dir=str(model_dir),
            scratch_roots=(local, scratch), protocol=protocol, venv=source, declared_version=tun.get('vllm_version'),
            gpu=gpu_name(report, group), base_env=os.environ,
            overrides={'tunables': a.tunables or None, 'protocol': a.protocol or None,
                       'replica_tunables': per_replica[i] if i < len(per_replica) else None})
        launches.append({**describe(argv, env), 'gpus': group, 'port': port, 'engine_cache': seeded,
                         'key': key, 'facts': facts, 'engine': str(engine), 'log': str(vlog), 'numerics': rec,
                         **({'venv_staged': venv_rec} if venv_rec else {})})
    host = socket.getfqdn()
    if gate is not None:
        # Admission gate (aa_evals/qwenserve/gate.py) in front of each replica: clients get the gate's port. With
        # tunable gate_pool (#110), one gate fronts every replica of the job (one URL, shared queues and KV share);
        # tunable gate_policy (JSON object) overrides the profile's gate policy, e.g. {"client_share": true}.
        policy = {**gate.get('policy', {}), **(tunables.get('gate_policy') or {})}
        fronts = [ports] if tunables.get('gate_pool') and len(ports) > 1 else [[p] for p in ports]
        for i, ups in enumerate(fronts):
            gport = consecutive_ports(1)
            argv = [PYTHON, '-u', '-m', 'aa_evals.qwenserve.gate',
                    '--upstream', ','.join(f'http://127.0.0.1:{p}' for p in ups),
                    '--port', str(gport), '--max-inflight', str(tunables.get('max_seqs', 128)),
                    '--policy', json.dumps(policy), '--log', str(scratch / f'gate-r{i}.jsonl'),
                    '--status', str(scratch / f'gate-r{i}.status.json'),
                    '--vllm-log', ','.join(launches[ports.index(p)]['log'] for p in ups)]
            gates.append((subprocess.Popen(argv, cwd=str(TS_ROOT_CODE), stdout=(scratch / f'gate-r{i}.out').open('ab'),
                                           stderr=subprocess.STDOUT, start_new_session=True), gport))
    public = [g for _, g in gates] if gates else ports
    job_id = os.environ.get('SERVE_JOB_ID')
    served = served_identity(teacher, [x['numerics'] for x in launches])
    # Per job (the home's launch.json is overwritten by the next start of this name): numerics.<job>.json keeps it
    record = {'job_id': job_id, 'teacher': a.teacher, 'pipeline': pipeline, 'host': host, 'time': time.time(), **served,
              'replicas': [x['numerics'] for x in launches]}
    atomic(home / f'numerics.{job_id or f"pid{os.getpid()}"}.json', record)
    # the first engine's command at the top level as well (task-synthesis' launch.json format): readers of
    # launch.json['argv'] (task_synthesis/pipeline/llm.py Endpoint._dp_size, m5/gdpval/llm.py) pin sessions to the DP
    # ranks of a dp_internal server from it; with only replicas[].argv they fell back to no pinning
    first = {k: launches[0][k] for k in ('argv', 'env', 'argv_json') if k in launches[0]} if launches else {}
    atomic(home / 'launch.json', {**first,
                                  'replicas': [{k: v for k, v in x.items() if k != 'facts'} for x in launches],
                                  'gpus': gpus, 'ports': ports, 'public_ports': public, 'gate': gate, 'host': host,
                                  'teacher': a.teacher, 'pipeline': pipeline, 'checkpoint': teacher['checkpoint'],
                                  'protocol': protocol, 'staged': staged, 'time': time.time(), 'job_id': job_id,
                                  **served})
    log(home, f'vLLM started on {host}: {len(procs)} process(es), GPUs {groups}, ports {ports}'
              + (f', gates {public}' if gates else ''))
    healthy_at, last_copy, copier, last_sources = None, 0.0, None, 0.0
    try:
        while not stopping:
            dead = [i for i, p in enumerate(procs) if p.poll() is not None]
            dead += [f'gate{i}' for i, (g, _) in enumerate(gates) if g.poll() is not None]
            if dead:
                log(home, f'exited: {dead} ' + str([p.returncode for p in procs]))
                break
            if time.time() - last_copy > 60 and not (copier and copier.is_alive()):
                # In a thread: a slow /data (over its byte quota, 2026-11-02) must never delay endpoints.json, which
                # clients treat as dead after 120 s.
                def copy_logs():
                    for i, x in enumerate(launches):
                        shutil.copyfile(x['log'], home / ('vllm.log' if i == 0 else f'vllm-r{i}.log'))
                    for i in range(len(gates)):
                        if (scratch / f'gate-r{i}.status.json').exists():
                            shutil.copyfile(scratch / f'gate-r{i}.status.json', home / f'gate-r{i}.status.json')
                copier = threading.Thread(target=copy_logs, daemon=True)
                copier.start()
                last_copy = time.time()
            if healthy_at is not None and time.time() - last_sources > SOURCES_SEC:
                last_sources = time.time()
                threading.Thread(target=record_prompt_sources, args=(home, ports), daemon=True).start()
            if healthy_at is None:
                try:
                    for p in ports + [g for _, g in gates]:
                        request(f'http://127.0.0.1:{p}/v1/models', timeout=3)
                except OSError:
                    if time.time() - t0 > serving.get('startup_timeout_sec', 5400):
                        log(home, 'startup timeout')
                        break
                    time.sleep(10)
                    continue
                for p in public + (ports if len(public) < len(ports) else []):   # a pool: each engine as well
                    run_probe(home, f'http://127.0.0.1:{p}/v1', teacher)
                healthy_at = time.time()
                log(home, f'healthy after {healthy_at - t0:.0f}s')
                for x in launches:
                    threading.Thread(target=lambda x=x: log(home, 'engine cache publish: ' + str(engine_cache.publish(
                        Path(x['engine']) / 'cache', x['key'], x['facts'], scratch,
                        {'host': host, 'engine': f'tasksynth-{a.name}'}))), daemon=True).start()
            atomic(home / 'endpoints.json', {'served_name': teacher['served_name'], 'teacher': a.teacher,
                                             'checkpoint': teacher['checkpoint'],
                                             'urls': [f'http://{host}:{p}/v1' for p in public],
                                             'engine_urls': [f'http://{host}:{p}/v1' for p in ports],
                                             # pi_eval holds new attempts while a gate queues new conversations
                                             'gate': gate is not None,
                                             'gate_status_urls': [f'http://{host}:{g}/gate/status' for _, g in gates],
                                             'healthy_at': healthy_at, 'updated_at': time.time(),
                                             'job_id': job_id,
                                             # what a result row needs to tell recipes under one served_name apart
                                             **served,
                                             'request': teacher.get('request', {})})
            time.sleep(30)
    finally:
        for g, _ in gates:
            stop_process(g)
        for p in procs:
            stop_process(p)
        for i, x in enumerate(launches):
            shutil.copyfile(x['log'], home / ('vllm.log' if i == 0 else f'vllm-r{i}.log'))
        for i in range(len(gates)):
            for name in (f'gate-r{i}.jsonl', f'gate-r{i}.status.json'):
                if (scratch / name).exists():
                    shutil.copyfile(scratch / name, home / name)
        (home / 'endpoints.json').unlink(missing_ok=True)
        log(home, 'stopped')
    sys.exit(0 if stopping else 1)


SOURCES_SEC = 300
SOURCE_RE = re.compile(r'^vllm:prompt_tokens_by_source_total\{[^}]*source="([a-z_]+)"[^}]*\} ([0-9.e+]+)', re.M)
GEN_RE = re.compile(r'^vllm:generation_tokens_total\{[^}]*\} ([0-9.e+]+)', re.M)


def prompt_sources(texts):
    """Cumulative prompt tokens by source (local_compute, local_cache_hit = GPU prefix cache, external_kv_transfer =
    the CPU KV tier) and generated tokens, summed over the /metrics texts of a job's API ports."""
    sources, gen = {}, 0.0
    for txt in texts:
        for name, value in SOURCE_RE.findall(txt):
            sources[name] = sources.get(name, 0.0) + float(value)
        gen += sum(float(v) for v in GEN_RE.findall(txt))
    return {'sources': sources, 'generation_tokens': gen}


def record_prompt_sources(home, ports):
    """Every SOURCES_SEC once healthy: prompt-token sources and generated tokens of every engine, appended to
    <home>/prompt-sources.<job>.jsonl, so whether a CPU KV tier earns its RAM on a workload can be read off any server
    (task synthesis #242/#246: the DeepSeek tier served 0.0% of one-shot prompt tokens). Best effort."""
    import urllib.request
    texts, failed = [], []
    for p in ports:   # one port that does not answer must not drop the others
        try:
            texts.append(urllib.request.urlopen(f'http://127.0.0.1:{p}/metrics', timeout=20).read().decode())
        except Exception:  # noqa: BLE001
            failed.append(p)
    if not texts:
        return
    try:
        with open(home / f'prompt-sources.{os.environ.get("SERVE_JOB_ID", "local")}.jsonl', 'a') as f:
            f.write(json.dumps({'time': time.time(), **prompt_sources(texts),
                                **({'ports_failed': failed} if failed else {})}) + '\n')
    except OSError:
        pass


def run_probe(home, url, teacher):
    """The startup text + tool probes (synth.probe). Fatal unless the spec sets serving.probe = "log" (models under
    test: a student's answer to the bare probe prompts is model behavior, not a serving fault); then the outcome goes
    to <home>/probe.jsonl and the server stays up."""
    if teacher['serving'].get('probe', 'fatal') != 'log':
        return probe(url, teacher)
    rec = {'time': time.time(), 'url': url}
    try:
        res = probe(url, teacher)
        rec.update(ok=True, text=res['text']['choices'][0]['message'], tool=res['tool']['choices'][0]['message'])
    except Exception as e:   # noqa: BLE001 -- logged, never fatal for these specs
        rec.update(ok=False, error=repr(e)[:4000])
    with (home / 'probe.jsonl').open('a') as f:
        f.write(json.dumps(rec, default=str) + '\n')
    log(home, f'startup probe on {url}: {"ok" if rec["ok"] else "FAILED (non-fatal, serving.probe = log)"}')
    return rec


# Memory safety net (#378): a server held for memory (HoldReasonCode 34, cgroup at its limit with nothing left to
# reclaim) is released by condor and restarts with MEMORY_STEP more of its base request per such hold, at most
# MEMORY_RELEASES times: a too-tight shape costs a restart (new node, ~2-15 min start-up; clients wait on a stale
# endpoints.json), not a dead model. Other holds (user, policy) are left alone.
# The constants are tbshared.holds' (the shared policy, moved there unchanged from this launcher, #415).
MEMORY_HOLDS, MEMORY_STEP = holds.MEMORY_HOLDS, holds.MEMORY_STEP
MEMORY_RELEASES, MEMORY_RELEASE = holds.MEMORY_RELEASES, holds.MEMORY_RELEASE


def memory_gib(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE):
    """The spec's RAM for a job of `gpus` GPUs: serving_resources.memory_gib (whole job) or memory_gib_per_gpu x gpus;
    the pipeline's profile memory_gib overrides both (#445: task-synthesis' copy ignored memory_gib_per_gpu)."""
    res = teacher.get('serving_resources', {})
    return (profile_for(slug, pipeline).get('memory_gib') or res.get('memory_gib')
            or res.get('memory_gib_per_gpu', 256) * gpus)


def cpus(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE):
    """The spec's CPUs for a job of `gpus` GPUs: serving_resources.cpus_per_gpu x gpus (default 8), unless the
    pipeline's profile names cpus."""
    return profile_for(slug, pipeline).get('cpus') or teacher.get('serving_resources', {}).get('cpus_per_gpu', 8) * gpus


def extra_engine_cpus(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE, tunables_json=''):
    """CPUs a job adds for engines beyond one per GPU: serving_resources.cpus_per_extra_engine (per GPU, default 0) x
    (k - 1) x gpus, k = engines_per_gpu as serve() plans it (spec, profile and --tunables). A spec sized for one engine
    per GPU (snowball-eval since #531) keeps that many cores for each extra engine; 0 for specs without the key."""
    per = float(teacher.get('serving_resources', {}).get('cpus_per_extra_engine') or 0)
    if not per or teacher['serving'].get('mode', 'replicas') != 'replicas':   # k > 1 only in replicas mode
        return 0
    taken = [0]

    def ports(n):   # placeholder API ports (distinct, as a plan needs): the plan's shape only
        taken[0] += n
        return taken[0] - n
    k = engine_plan(slug, teacher, list(range(gpus)), pipeline=pipeline, tunables_json=tunables_json, ports=ports)['k']
    return int(round(per * (k - 1) * gpus))


def replica_launch(teacher, tun, protocol, *, gpus, port, model_dir, rank_ports, scratch, kernel_provider=None,
                   staged_venvs=None):
    """(argv, env, source venv, its node-local copy or None, unstage) of one engine. With tunable stage_venv and a copy
    of its venv in `staged_venvs` (source -> (path, record)), the engine runs the copy everywhere the venv appears:
    entrypoint, PATH (a spec's literal PATH too), a cu13-wheel CUDA_HOME built over the copy. `unstage` maps a string
    back to the source venv: the numerics are recorded with it (same files, so the same numerics_sha256)."""
    source = tun.get('venv') or teacher['runtime']['venv']
    copy = (staged_venvs or {}).get(source, (None, None))[0] if tun.get('stage_venv') else None
    r_teacher, r_tun = teacher, tun
    if copy:
        r_teacher = {**teacher, 'runtime': {**teacher['runtime'], 'venv': str(copy)}}
        r_tun = {**tun, 'venv': str(copy)} if tun.get('venv') else tun
    argv = replica_command(r_teacher, protocol, port=port, model_dir=model_dir, tunables=r_tun, rank_ports=rank_ports)
    env = replica_env(r_teacher, gpus=gpus, scratch=scratch, tunables=r_tun)
    if kernel_provider:   # as synth.serve: the spec's prebuilt kernels first on PYTHONPATH
        env['PYTHONPATH'] = ':'.join(filter(None, [str(kernel_provider), env.get('PYTHONPATH')]))
    if not copy:
        return argv, env, source, None, (lambda x: x)
    argv = [swap_path(a, source, copy) for a in argv]
    env = {k: swap_path(v, source, copy) for k, v in env.items()}
    return argv, env, source, copy, (lambda x, c=str(copy): swap_path(x, c, source))


def venvs_to_stage(plan, teacher) -> list:
    """The venvs a job copies to the node: those of its engines with tunable stage_venv (job-wide, a profile's, or a
    replica's own) - each engine's tunable venv, else the spec's."""
    per = plan['per_replica']
    out = set()
    for i in range(len(plan['groups'])):
        tun = {**plan['tunables'], **(per[i] if i < len(per) else {})}
        if tun.get('stage_venv'):
            out.add(tun.get('venv') or teacher['runtime']['venv'])
    return sorted(out)


def placeholder_plan(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE, tunables_json=''):
    """engine_plan on placeholder GPUs and API ports: a submit's view of the job's shape."""
    taken = [0]

    def ports(n):
        taken[0] += n
        return taken[0] - n
    return engine_plan(slug, teacher, list(range(gpus)), pipeline=pipeline, tunables_json=tunables_json, ports=ports)


def staged_venv_gib(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE, tunables_json=''):
    """Node-local disk the job's staged venvs take (their tars' sizes; tunable stage_venv), 0 without it."""
    return sum(venv_stage.tar_gib(v) for v in venvs_to_stage(placeholder_plan(teacher, slug, gpus, pipeline,
                                                                               tunables_json), teacher))


def kv_tier_gib(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE, tunables_json=''):
    """GiB of pinned host RAM the job's CPU KV tiers take: kv_offloading_size (spec, profile, --tunables and a
    replica's replica_tunables, as serve() merges them) on every DP rank of every engine (vLLM's size is per engine
    core, summed over its TP ranks). 0 when the spec has no kv_offloading_size optional arg or the size is 0."""
    if 'kv_offloading_size' not in teacher['serving'].get('optional_args', {}):
        return 0.0
    plan = placeholder_plan(teacher, slug, gpus, pipeline, tunables_json)
    per = plan['per_replica']
    return float(sum(float({**plan['tunables'], **(per[i] if i < len(per) else {})}.get('kv_offloading_size') or 0)
                     * int(extra.get('dp_size', 1)) for i, extra in enumerate(plan['extras'])))


def derived_memory_gib(teacher, slug, gpus, pipeline=DEFAULT_PIPELINE, tunables_json=''):
    """The job's RAM from the spec and its effective CPU KV tier -> (GiB, tier): the spec's (or profile's) RAM
    (memory_gib) is sized for the tier the spec and profile set, on the GPUs it is sized for (a whole-job memory_gib:
    the spec's default_gpus, else 4; per-GPU RAM: the job's), so the job's own tier (--tunables, replica_tunables, or
    the spec's on another GPU count) moves it by the difference (#451 ws4: m5l-gdpval ran a 48 GiB tier on RAM sized
    for Super's 200 GiB one). tier = {'spec': GiB, 'spec_gpus': GPUs it is counted on, 'job': GiB on `gpus`}."""
    res = teacher.get('serving_resources', {})
    whole_job = bool(profile_for(slug, pipeline).get('memory_gib') or res.get('memory_gib'))
    base = int(res.get('default_gpus', 4)) if whole_job else gpus
    spec = kv_tier_gib(teacher, slug, base, pipeline)
    job = kv_tier_gib(teacher, slug, gpus, pipeline, tunables_json)
    return memory_gib(teacher, slug, gpus, pipeline) + job - spec, {'spec': spec, 'spec_gpus': base, 'job': job}


SHAPE_LIMIT = 1.5   # the same rule as pi_eval's worker shapes (#451 workstream 4)


def spec_per_gpu(teacher, slug, pipeline=DEFAULT_PIPELINE):
    """(CPUs, GiB) per GPU the spec and the pipeline's profile size a server for: a profile's cpus / memory_gib and a
    spec's whole-job memory_gib are per job, sized for the spec's GPU count (serving_resources.default_gpus, else 4);
    cpus_per_gpu / memory_gib_per_gpu are per GPU."""
    res = teacher.get('serving_resources', {})
    profile = profile_for(slug, pipeline)
    base = int(res.get('default_gpus', 4))
    c = profile['cpus'] / base if profile.get('cpus') else res.get('cpus_per_gpu', 8)
    if profile.get('memory_gib'):
        m = profile['memory_gib'] / base
    elif res.get('memory_gib'):
        m = res['memory_gib'] / base
    else:
        m = res.get('memory_gib_per_gpu', 256)
    return c, m


def shape_check(teacher, slug, gpus, request_cpus, request_gib, pipeline=DEFAULT_PIPELINE, tier=None,
                extra_cpus=0) -> dict:
    """The job's request per GPU against the expected shape: spec_per_gpu, its RAM moved by the job's CPU KV tier per
    GPU less the spec's (`tier` from derived_memory_gib) and its CPUs by the extra engines' (`extra_cpus`,
    extra_engine_cpus). ok unless CPUs or RAM per GPU is over SHAPE_LIMIT x the expected (#451 ws4)."""
    sc, sm = spec_per_gpu(teacher, slug, pipeline)
    sc += extra_cpus / gpus
    if tier:
        sm += tier['job'] / gpus - tier['spec'] / tier['spec_gpus']
    rc, rm = request_cpus / gpus, request_gib / gpus
    ratio = {'cpus': round(rc / sc, 3) if sc else None, 'memory_gib': round(rm / sm, 3) if sm else None}
    return {'requested_per_gpu': {'cpus': round(rc, 2), 'memory_gib': round(rm, 2)},   # before the fill
            'expected_per_gpu': {'cpus': round(sc, 2), 'memory_gib': round(sm, 2)}, 'ratio': ratio, 'limit': SHAPE_LIMIT,
            'ok': all(r is None or r <= SHAPE_LIMIT for r in ratio.values())}


def memory_request(gib):
    """request_memory (MB) as an expression: the base, plus MEMORY_STEP of it per memory hold so far."""
    return holds.memory_request(gib)


def condor_arguments(argv):
    """HTCondor new-style `arguments`: the whole line in double quotes (inner ones doubled), each argument in single
    quotes (inner ones doubled), so JSON like {"mtp": 0} survives."""
    return '"' + ' '.join("'" + x.replace("'", "''").replace('"', '""') + "'" for x in argv) + '"'


def submit(a):
    teacher = load_teacher(a.teacher, checkpoint_overrides(a))
    check_checkpoint(teacher)
    pipeline = getattr(a, 'pipeline', None) or DEFAULT_PIPELINE
    profile_for(a.teacher, pipeline)   # an unknown pipeline fails before anything is written
    res = teacher.get('serving_resources', {})
    gpus = a.gpus or int(res.get('default_gpus', 4))
    # the derived shape: the spec's CPUs, and its RAM moved by the job's CPU KV tier difference (a --tunables
    # kv_offloading_size smaller than the spec's needs less pinned host RAM; #451 ws4, coordinator: size it)
    try:   # a plan the job could not start (bad --tunables JSON, GPUs that do not split) fails here, readably
        json.loads(a.tunables or '{}')
        derived_gib, tier = derived_memory_gib(teacher, a.teacher, gpus, pipeline, a.tunables)
        extra_cpus = extra_engine_cpus(teacher, a.teacher, gpus, pipeline, a.tunables)
        venv_gib = staged_venv_gib(teacher, a.teacher, gpus, pipeline, a.tunables)
    except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as e:
        raise SystemExit(f'{a.name}: cannot plan {a.teacher} on {gpus} GPU(s) with --tunables '
                         f"{a.tunables or '{}'}: {type(e).__name__}: {e}")
    derived_cpus = cpus(teacher, a.teacher, gpus, pipeline) + extra_cpus
    gib = a.memory_gib or derived_gib
    request_cpus = a.cpus or derived_cpus
    # the requested shape per GPU (before the fill, which only fills to the binding resource) against the derived one:
    # over SHAPE_LIMIT x either one refuses the submit (exit 2) before anything is written, unless --allow-shape;
    # recorded in job.json. It catches oversized --cpus / --memory-gib and whole-job-RAM specs on fewer GPUs than they
    # are sized for.
    check = {'requested': {'cpus': request_cpus, 'memory_gib': gib},
             'derived': {'cpus': derived_cpus, 'memory_gib': derived_gib}, 'kv_tier_gib': tier,
             **({'extra_engine_cpus': extra_cpus} if extra_cpus else {}),
             **shape_check(teacher, a.teacher, gpus, request_cpus, gib, pipeline, tier, extra_cpus),
             'allowed': getattr(a, 'allow_shape', False) is True}
    if not check['ok'] and not check['allowed']:
        over = [k for k, r in check['ratio'].items() if r is not None and r > SHAPE_LIMIT]
        rq, sp = check['requested_per_gpu'], check['expected_per_gpu']
        # short (launchers keep the tail of stderr): what is over, by how much, and the way out
        print(f"{a.name}: {'/'.join(over)} per GPU over {SHAPE_LIMIT}x expected: {rq['cpus']:g}c/{rq['memory_gib']:g}G vs "
              f"{sp['cpus']:g}c/{sp['memory_gib']:g}G ({a.teacher}, {gpus} GPU); --allow-shape to submit", file=sys.stderr)
        sys.exit(2)
    home = ROOT / a.name
    home.mkdir(parents=True, exist_ok=True)
    job = read(home / 'job.json')
    if job and not a.force and not a.dry_run:
        raise SystemExit(f'{home}/job.json exists ({job["job_id"]}); condor_rm it and pass --force')
    defaults = PIPELINE_SUBMIT[pipeline]
    fill = defaults['fill'] if getattr(a, 'fill', None) is None else a.fill
    net = defaults['memory_net']
    code = frozen_checkout()
    # omitted --gpus / --bid: the spec's serving_resources.default_gpus / default_bid, else 4 GPUs at bid 63 as before
    a.bid = a.bid or int(res.get('default_bid', 63))
    env = {'USER': os.environ.get('USER', 'user'), 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
           'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(code), 'SERVE_JOB_ID': '$(Cluster).$(Process)', **PROXY_ENV,
           # the job writes where this submit did (a canary's own registry stays its own)
           **({tbserving.ROOT_ENV: str(ROOT)} if ROOT != (TS_ROOT / 'serving').absolute() else {})}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    exclude = sorted(set(res.get('exclude_nodes', []) + bad_nodes() + [x for x in a.exclude_nodes.split(',') if x]))
    cuda_cap = a.cuda_cap or res.get('cuda_capability', '10.0')   # the spec's GPU type unless overridden
    if 'cuda_capabilities' in res and cuda_cap not in res['cuda_capabilities']:   # e.g. kernels built for one arch
        raise SystemExit(f'{a.teacher} serves on CUDA capability {", ".join(res["cuda_capabilities"])} only, not {cuda_cap}')
    # /dev/shm: the job wrapper mounts it at the job ad's DevShmSize (MiB), else half the slot's memory
    dev_shm = profile_for(a.teacher, pipeline).get('dev_shm_gib') or res.get('dev_shm_gib')
    sub = home / ('job.sub.dry' if a.dry_run else 'job.sub')   # a dry run never touches a live job's receipt
    # with the memory net: request_memory grows per memory hold and periodic_release lets condor restart the job
    memory_lines = holds.submit_lines(gib) if net else f'request_memory = {int(gib * 1024)}\n'
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = {condor_arguments(['-u', '-m', 'task_synthesis.serve.serve_teacher', 'run', '--teacher', a.teacher, '--name', a.name]
                              + (['--pipeline', pipeline] if pipeline != DEFAULT_PIPELINE else [])
                              + (['--tunables', a.tunables] if a.tunables else [])
                              + (['--protocol', a.protocol] if a.protocol else [])
                              + (['--checkpoint', teacher['checkpoint']] if a.checkpoint else []))}
initialdir = {code}
output = {home}/job.$(Cluster).out
error = {home}/job.$(Cluster).err
log = {home}/job.condor.log
request_cpus = {request_cpus}
{memory_lines}request_disk = {int(res.get('disk_gib', 500)) + math.ceil(venv_gib)}G
request_gpus = {gpus}
+JobBatchName = "tasksynth-serve-{a.name}"
+MaxRuntime = {a.hours * 3600}
{'+BypassLXCfs = "true"' + chr(10) if res.get('bypass_lxcfs') or a.bypass_lxcfs else ''}{f'+DevShmSize = {int(dev_shm * 1024)}' + chr(10) if dev_shm else ''}getenv = false
environment = "{environment}"
requirements = {requirements(exclude, True, cuda_cap)}
queue
''')
    # tbshared 1.5 fill (the user's rule; task-synthesis' default): CPUs and RAM at the binding level within the GPU
    # class's per-GPU share, so a filled server never strands GPUs; unchanged when the class is unknown. The memory
    # net's expression is filled at its base. Decisions go to fill.jsonl next to job.sub (not for dry runs). Without
    # fill the submit file is left exactly as written (aa-serving's behavior).
    if fill:
        body, n = condor_cost.split_queue(sub.read_text())
        body = condor_cost.fill_plan(body, n, None, exclude, fill=True, where=str(sub), record=not a.dry_run,
                                     pipeline=pipeline)[0]   # the record names --pipeline, not this package
        sub.write_text(body + f'queue {n}\n')
    shape = condor_cost.parse_shape(sub.read_text())[:2]
    if a.dry_run:   # the submit file is written; condor_submit -dry-run checks it without queueing anything
        r = subprocess.run(['condor_submit_bid', str(a.bid), '-dry-run', '-', str(sub)], capture_output=True, text=True)
        print(sub.read_text() + (r.stdout + r.stderr).strip()[-2000:])
        return
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    job_id = re.search(r'cluster (\d+)', r.stdout).group(1) + '.0'
    # the submit-time overrides (the job's effective numerics: <home>/numerics.<job>.json once it starts)
    atomic(home / 'job.json', {'job_id': job_id, 'teacher': a.teacher, 'checkpoint': teacher['checkpoint'],
                               'pipeline': pipeline, 'gpus': gpus, 'bid': a.bid, 'cuda_cap': cuda_cap,
                               'cpus': shape[0], 'memory_gib': shape[1], 'fill': fill,
                               # whether condor itself releases a memory hold (tbshared.holds.remove_held(has_net=))
                               'memory_net': net,
                               'shape_check': check,
                               'code': str(code), 'time': time.time(),
                               'overrides': {'tunables': a.tunables or None, 'protocol': a.protocol or None,
                                             'cpus': a.cpus or None, 'memory_gib': a.memory_gib or None,
                                             'cuda_cap': a.cuda_cap or None, 'hours': a.hours}})
    print(job_id)


def registry(a):
    """status / wait / release of a server by name (tbshared.serving; ROOT is serve_teacher's registry)."""
    serving = tbserving
    if a.op == 'status':
        print(json.dumps(serving.state(a.name, ROOT)))
        return 0
    if a.op == 'wait':
        e = serving.wait_ready(a.name, a.timeout, ROOT)
        print(json.dumps({'ready': e is not None, **({'urls': e.get('urls'), 'job_id': e.get('job_id')} if e else
                                                     serving.state(a.name, ROOT))}))
        return 0 if e is not None else 1
    out = serving.release(a.name, a.reason, ROOT)
    print(json.dumps(out))
    return 0 if out['action'] == 'removed' or out.get('why') == 'not in the queue' else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    for op in ('status', 'wait', 'release'):
        p = sub.add_parser(op)
        p.add_argument('--name', required=True)
        if op == 'wait':
            p.add_argument('--timeout', type=float, default=7200, help='seconds (default %(default)s: start-up limit '
                           '5,400 s plus queue time)')
        if op == 'release':
            p.add_argument('--reason', required=True)
    for op in ('run', 'submit'):
        p = sub.add_parser(op)
        p.add_argument('--teacher', required=True)
        p.add_argument('--name', required=True)
        p.add_argument('--tunables', default='', help='JSON tunables over the spec and PROFILE, e.g. {"kv_offloading_size": 80}')
        p.add_argument('--protocol', default='', help='JSON protocol over the PROFILE\'s, e.g. '
                       '{"context_tokens": 262144, "max_output_tokens": 262144} (max_model_len, output cap)')
        p.add_argument('--checkpoint', default='', help="model directory in place of the spec's checkpoint (every "
                       "{checkpoint} in the spec follows it); checked against the spec's [checkpoint_checks]")
        p.add_argument('--pipeline', default=DEFAULT_PIPELINE, choices=PIPELINES,
                       help='profile set and submit defaults (PIPELINE_PROFILES, PIPELINE_SUBMIT); default: %(default)s')
        if op == 'submit':
            p.add_argument('--gpus', type=int, default=None, help="default: the spec's serving_resources.default_gpus, else 4")
            p.add_argument('--bid', type=int, default=None, help="default: the spec's serving_resources.default_bid, else 63")
            p.add_argument('--hours', type=int, default=48)
            p.add_argument('--force', action='store_true')
            p.add_argument('--dry-run', action='store_true', help='write and check the submit file, queue nothing')
            p.add_argument('--bypass-lxcfs', action='store_true', help='+BypassLXCfs on the job (default: the spec\'s '
                           'serving_resources.bypass_lxcfs)')
            p.add_argument('--exclude-nodes', default='', help='comma-separated nodes to exclude on top of the '
                           "spec's exclude_nodes and the shared bad nodes (e.g. a campaign's excluded-nodes list)")
            p.add_argument('--cpus', type=int, default=0, help="CPUs for the job; default: the spec's cpus_per_gpu x gpus, plus its cpus_per_extra_engine per GPU for "
                           "each engine beyond one per GPU (engines_per_gpu)")
            p.add_argument('--memory-gib', type=float, default=0, help="RAM for the job (GiB) instead of the spec's "
                           "shape, e.g. for A/B arms with a CPU KV tier; the memory-hold safety net still applies where "
                           "the pipeline has it (PIPELINE_SUBMIT memory_net)")
            p.add_argument('--fill', dest='fill', action='store_true', default=None,
                           help="request CPUs / RAM at the binding level within the GPU class (tbshared condor_cost."
                                "fill_plan); default: the pipeline's (PIPELINE_SUBMIT)")
            p.add_argument('--no-fill', dest='fill', action='store_false',
                           help="request the spec's CPUs / RAM as given")
            p.add_argument('--allow-shape', action='store_true',
                           help=f"submit even if CPUs or RAM per GPU is over {SHAPE_LIMIT}x the derived shape's (the spec's, its RAM "
                                "moved by the job's CPU KV tier; recorded in job.json)")
            p.add_argument('--cuda-cap', default='', help="GPU CUDA capability: 10.0 = B200, 9.0 = H100; default: the "
                                                          "spec's serving_resources.cuda_capability, else 10.0")
    a = ap.parse_args()
    if a.op in ('status', 'wait', 'release'):
        sys.exit(registry(a))
    (run if a.op == 'run' else submit)(a)


if __name__ == '__main__':
    main()
