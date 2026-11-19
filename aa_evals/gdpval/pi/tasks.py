"""GDPval-AA tasks as Harbor tasks for the TB 2.1 Pi harness (#348).

  python -m aa_evals.gdpval.pi.tasks build [--out DIR]       # once: the bundle tar + its manifest (CPU, ~3 GB)

Each of the 220 GDPval tasks becomes a Harbor task that `aa_evals.common.pi_eval` runs like a TB 2.1 task:
- environment: one shared image for every task, `docker_image = BASE_REF`: the GDPval sandbox SIF's root filesystem
  (`/data/aai/aa_evals/sif/gdpval-v1.sif`, AA's package lists), put into harbor_ns's shared base cache by
  `sandbox/build_ns_base.sh`. No per-task image.
- inputs (`pi_eval/task_inputs.py`): the task's reference files at `/home/user/<name>` (same names, same Office
  package patch as the Stirrup path), an empty `/home/user/outputs/`, and for the 3 tasks that need them the
  generic dependency caches of `gdpval/runtime_cache.py` (read-only payload at /opt/aa-runtime-cache, npm cache in
  the home, offline npm/pip/terraform settings as config files instead of per-command environment).
- instruction: AA's system and task prompts without the finish-tool and step-budget text (`pi_instruction`):
  deliverables are the files in `/home/user/outputs/` when the episode ends.
- artifacts: `/home/user/outputs` is collected by Harbor after the agent (also after a time limit or a context-limit
  end); `tests/test.sh` only records how many files are there (reward 1 when at least one; not the GDPval score).
  The GDPval module imports the deliverables, renders and judges them as before (`pi/collect.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from aa_evals.common.pi_eval import task_inputs
from aa_evals.gdpval import aa_prompts
from aa_evals.gdpval import runtime_cache as RC
from aa_evals.gdpval import tasks as T
from aa_evals.gdpval.office_fix import fix_file

SIF_SHA256 = '79079522716486d27779ba4cef117859a999f661f702c08513dfce27fabb8d07'
BASE_REF = f'aa-evals.local/gdpval-sandbox:v1-{SIF_SHA256[:12]}'
BUNDLE_VERSION = 'gdpval-pi-v1'
OUTPUT_DIR = '/home/user/outputs'
OUT = Path('/data/aai/aa_evals/data/gdpval/pi_bundle')
MEMORY_MB = 16384   # the Stirrup path's 16 GB per-command cap; here the sandbox's (HARBOR_NS_MEMCAP=1)
CPUS = 2
# Not in AA's prompt (accepted, #348): without compaction the episode ends when the window fills, so the agent is told.
END_DISCLOSURE = ('Your session can also end without warning (for example when the conversation reaches its length '
                  'limit), and then whatever is in that directory at that moment is your submission, so save each '
                  'deliverable there as soon as you produce it.')

TASK_TOML = '''schema_version = "1.1"
artifacts = ["{outputs}"]

[metadata]
benchmark = "gdpval"
bundle_version = "{version}"
task_id = "{task_id}"
sector = {sector}
occupation = {occupation}

[verifier]
timeout_sec = 300.0

[agent]
timeout_sec = 21600.0

[environment]
build_timeout_sec = 1800.0
docker_image = "{base}"
cpus = {cpus}
memory_mb = {memory}
storage_mb = 20480
gpus = 0
allow_internet = false
'''

DOCKERFILE = f'''# Not built while the shared image exists (task.toml docker_image is used first); kept as harbor_ns's fallback.
FROM {BASE_REF}
WORKDIR /home/user
'''

# Records what the agent left in the output directory. The deliverables themselves are Harbor artifacts (collected
# before this runs); the GDPval score comes from the judge, not from this reward.
TEST_SH = f'''#!/bin/bash
mkdir -p /logs/verifier
n=$(find {OUTPUT_DIR} -type f 2>/dev/null | wc -l)
find {OUTPUT_DIR} -type f -printf '%s\\t%P\\n' 2>/dev/null | head -1000 > /logs/verifier/outputs.tsv
if [ "$n" -gt 0 ]; then r=1; s=passed; else r=0; s=failed; fi
echo $r > /logs/verifier/reward.txt
# a CTRF report, so pi_eval's verifier wrapper (eval_q35 VERIFIER_WRAPPER) sees a finished check and does not retry
printf '{{"results":{{"tool":{{"name":"gdpval-outputs"}},"summary":{{"tests":1,"passed":%d,"failed":%d}},"tests":[{{"name":"files in {OUTPUT_DIR}","status":"%s"}}]}}}}\n' $r $((1-r)) $s > /logs/verifier/ctrf.json
echo "files in {OUTPUT_DIR}: $n"
'''


def _cut(text: str, old: str, new: str = '') -> str:
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


def system_text() -> str:
    """AA's system prompt without the finish-tool and step-budget text (#348)."""
    s = aa_prompts.SYSTEM_PROMPT
    s = _cut(s, ' within 250 steps, then submit your work', '')
    s = _cut(s, 'When you are done, call the `finish` tool as your final step with:\n1. A brief summary of what you '
                'accomplished.\n2. Absolute paths to every deliverable file.\n\n', '')
    s = _cut(s, 'call the `abandon_task_finish` tool with a brief reason instead',
             'stop and say why in your final message instead of producing files')
    s = _cut(s, 'Do not use it to escape difficulty.', 'Do not do so to escape difficulty.')
    s = _cut(s, 'record them in your finish summary', 'record them in your final message')
    return s


def task_text(task: T.Task) -> str:
    """AA's task prompt (tasks.task_prompt) with the runtime notes of this harness and the output directory in
    place of the finish tool."""
    t = T.task_prompt(task)
    t = _cut(t, 'Use the `code_exec` tool to read, create, and modify files. Commands run as the non-root user `user` '
                '(UID 1000). Default working directory is `/home/user`.',
             'Use your tools to read, create, and modify files. Default working directory is `/home/user`.')
    t = _cut(t, 'no working directory, environment variable, or other shell state carries over from one call to '
                'the next.', 'no working directory, environment variable, or other shell state carries over from one '
                'shell command to the next.')
    t = _cut(t, '- Commands are terminated after 10 minutes. Keep them bounded,',
             '- Keep commands bounded,')
    start, end = t.index('## Completing Your Work'), t.index('## Task\n')
    t = t[:start] + (
        '## Completing Your Work\n\n'
        f'Save every required output file in `{OUTPUT_DIR}/`. When you finish, the files in that directory are your '
        'submission; nothing else is collected. ' + END_DISCLOSURE + '\n\n'
        '**Required:**\n'
        f'1. Each required output file saved in `{OUTPUT_DIR}/` (files, not folders)\n'
        '2. A brief summary of what you accomplished in your final message\n\n') + t[end:]
    return t


def pi_instruction(task: T.Task, extra_notice: str = '') -> str:
    return system_text() + '\n\n' + task_text(task) + extra_notice


def _runtime_inputs(task: T.Task, manifest: dict, root: Path) -> tuple[str, dict]:
    """The Stirrup path's dependency cache for the 3 tasks that need one (runtime_cache.prepare), as files."""
    entry = RC.checked_entry(manifest, task, SIF_SHA256)
    payload = Path(entry['payload_root'])
    shutil.copytree(payload, root / RC.MOUNT.lstrip('/'))
    shutil.copytree(payload / 'npm', root / 'home/user/.cache/aa-npm')
    npmrc = ('cache=/home/user/.cache/aa-npm\noffline=true\naudit=false\nfund=false\nupdate-notifier=false\n')
    (root / 'home/user/.npmrc').write_text(npmrc)
    (root / 'home/user/.local/etc').mkdir(parents=True, exist_ok=True)
    (root / 'home/user/.local/etc/npmrc').write_text(npmrc)   # npm's global config once node is in ~/.local
    (root / 'home/user/.config/pip').mkdir(parents=True, exist_ok=True)
    (root / 'home/user/.config/pip/pip.conf').write_text('[global]\nno-index = true\n')
    if any(p.startswith('terraform-providers/') for p in entry['files']):
        (root / 'home/user/.terraformrc').write_text(
            'provider_installation {\n  filesystem_mirror {\n    path = "' + RC.MOUNT +
            '/terraform-providers"\n  }\n}\ndisable_checkpoint = true\n')
    notice = ('\n\nEvaluator setup cached only generic dependencies required by this task. Internet remains disabled. '
              f'Read-only artifacts are at {RC.MOUNT}. To install the supplied Node/npm runtime locally, create '
              '/home/user/.local and unpack /opt/aa-runtime-cache/runtime/node.tar.xz there with one leading '
              'archive directory removed. Python standard-library tarfile/lzma can unpack this archive; the external '
              'xz command is absent. Ignore archive ownership and preserve executable modes when extracting. '
              '/home/user/.local/bin is on PATH. npm is configured for an attempt-local seeded cache and offline '
              'mode; install needed cached packages into your own project. '
              'Available pinned package selectors: ' + ', '.join(entry['requested']) + '. '
              'Other supplied runtime archives, if any, are under /opt/aa-runtime-cache/runtime. '
              'No task implementation, external data or credentials are supplied.')
    return notice, {'runtime_cache': {'manifest_sha256': RC.sha(RC.DEFAULT_MANIFEST),
                                      'cache_version': manifest['cache_version'], 'files': len(entry['files'])}}


def build_task(task: T.Task, dest: Path, rc_manifest: dict | None) -> dict:
    d = dest / task.task_id
    (d / 'environment').mkdir(parents=True)
    (d / 'tests').mkdir()
    inp = d / task_inputs.INPUTS
    (inp / OUTPUT_DIR.lstrip('/')).mkdir(parents=True)
    patched = []
    for src, target in task.ref_targets():
        p = inp / target.lstrip('/')
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, p)
        if fix_file(p):   # AA patched these files for LibreOffice (office_fix.py), as agent.run_attempt does
            patched.append(p.name)
    notice, extra = ('', {})
    if task.task_id in RC.REQUIRED_TASKS:
        notice, extra = _runtime_inputs(task, rc_manifest, inp)
    instruction = pi_instruction(task, notice)
    (d / 'instruction.md').write_text(instruction)
    q = json.dumps
    (d / 'task.toml').write_text(TASK_TOML.format(outputs=OUTPUT_DIR, version=BUNDLE_VERSION, task_id=task.task_id,
                                                  sector=q(task.sector), occupation=q(task.occupation),
                                                  base=BASE_REF, cpus=CPUS, memory=MEMORY_MB))
    (d / 'environment' / 'Dockerfile').write_text(DOCKERFILE)
    (d / 'tests' / 'test.sh').write_text(TEST_SH)
    (d / 'tests' / 'test.sh').chmod(0o755)
    m = task_inputs.write_manifest(d, {'office_patched': patched, **extra})
    return {'instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
            'n_refs': len(task.reference_files), 'n_input_files': len(m['files']), 'office_patched': patched,
            'original_agent_timeout_sec': None, **extra}


def build(out: Path) -> Path:
    tasks = T.load_tasks()
    rc_manifest = RC.load_manifest(RC.DEFAULT_MANIFEST, RC.sha(RC.DEFAULT_MANIFEST))
    out.mkdir(parents=True, exist_ok=True)
    tar = out / f'{BUNDLE_VERSION}.tar'
    with tempfile.TemporaryDirectory(dir=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') as tmp:
        root = Path(tmp) / 'tasks'
        root.mkdir()
        meta = {t.task_id: build_task(t, root, rc_manifest) for t in tasks}
        part = tar.with_suffix('.tar.partial')
        with tarfile.open(part, 'w') as tf:
            for p in sorted(Path(tmp).rglob('*')):
                tf.add(p, arcname=p.relative_to(tmp).as_posix(), recursive=False, filter=_norm)
        part.replace(tar)
    manifest = {'bundle_version': BUNDLE_VERSION, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'base_ref': BASE_REF, 'sif_sha256': SIF_SHA256, 'dataset_revision': T.REVISION,
                'parquet_sha256': T.parquet_sha(), 'output_dir': OUTPUT_DIR, 'tasks': meta,
                'sha256': task_inputs.sha256(tar)}
    tar.with_suffix('.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{tar}: {len(meta)} tasks, sha256 {manifest["sha256"]}')
    return tar


def _norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ''
    ti.mtime = 0
    return ti


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['build', 'show'])
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--task-id', default=None, help='show: print one task\'s instruction')
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        build(Path(a.out))
    else:
        tasks = {t.task_id: t for t in T.load_tasks()}
        print(pi_instruction(tasks[a.task_id or sorted(tasks)[0]]))


if __name__ == '__main__':
    main()
