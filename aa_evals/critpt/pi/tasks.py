"""CritPt challenges as Harbor tasks for the TB 2.1 Pi harness (#348).

  python -m aa_evals.critpt.pi.tasks build      # once: the bundle tar + manifest (seconds; no answers in it)
  python -m aa_evals.critpt.pi.tasks show [--problem Challenge_1_main]

Each of the 70 challenges becomes a Harbor task that `aa_evals.common.pi_eval` runs like a TB 2.1 task:
- environment: one shared image, `docker_image = BASE_REF`: the CritPt sandbox SIF (`critpt_py313_20261102.sif`,
  python:3.13-slim + numpy, scipy, sympy, mpmath, matplotlib) put into harbor_ns's base cache by
  `aa_evals/common/pi_eval/sif_base.py` (`build_ns_base.sh`), with the Stirrup sandbox's settings as its environment
  (HOME=/work, 4 BLAS threads, MPLBACKEND=Agg); working directory /work; 16 GB memory cap (HARBOR_NS_MEMCAP=1).
- instruction: the authors' one-step system prompt, then the Stirrup path's user prompt (prompts.user_prompt) with
  the finish tool and the turn budget replaced by the answer file: the filled-in code template in `/work/answer.py`
  when the episode ends is the submission (`ANSWER`).
- artifacts: `/work/answer.py` (Harbor collects it after the agent, also after a time or context limit);
  `tests/test.sh` only records whether it exists and parses (reward 1/0; not the CritPt score). The CritPt module
  imports the answer as the row's `answer_code`, then probes and grades it unchanged (`pi/run.py`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path

from aa_evals.critpt import data, prompts

SIF = data.DATA_ROOT / 'sif' / 'critpt_py313_20261102.sif'
SIF_SHA256 = 'a4d1c3614ab4cdc2dd2d15a03c903af69b8b8a4d2d398b60e7996fc1f020cf3e'
BASE_REF = f'aa-evals.local/critpt-sandbox:py313-20261102-{SIF_SHA256[:12]}'
BASE_ENV = {'HOME': '/work', 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': '/tmp/mpl', 'OMP_NUM_THREADS': '4',
            'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}   # harness.sandbox_argv
WORKDIR = '/work'
ANSWER = '/work/answer.py'
BUNDLE_VERSION = 'critpt-pi-v1'
OUT = data.DATA_ROOT / 'pi_bundle'
# Not in the Stirrup prompt (accepted, #348): without compaction the episode ends when the window fills.
END_DISCLOSURE = (f'Your session can end without warning (for example when the conversation reaches its length limit), '
                  f'and then `{ANSWER}` is graded as it is at that moment. Write it as soon as you have a candidate '
                  'answer and update it if you improve the answer.')

TASK_TOML = '''schema_version = "1.1"
artifacts = ["{answer}"]

[metadata]
benchmark = "critpt"
bundle_version = "{version}"
problem_id = "{pid}"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 21600.0

[environment]
build_timeout_sec = 1800.0
docker_image = "{base}"
cpus = 4
memory_mb = 16384
storage_mb = 10240
gpus = 0
allow_internet = false
'''

DOCKERFILE = f'''# Not built while the shared image exists (task.toml docker_image is used first); kept as harbor_ns's fallback.
FROM {BASE_REF}
WORKDIR {WORKDIR}
'''

TEST_SH = f'''#!/bin/bash
# Records whether the answer file exists and parses; the CritPt score comes from the consensus grader, not this.
mkdir -p /logs/verifier
if [ -f {ANSWER} ] && python -I -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" {ANSWER}; then
  r=1; s=passed; echo "answer parses"
else
  r=0; s=failed; echo "no parsable {ANSWER}"
fi
echo $r > /logs/verifier/reward.txt
# a CTRF report, so pi_eval's verifier wrapper (eval_q35 VERIFIER_WRAPPER) sees a finished check and does not retry
printf '{{"results":{{"tool":{{"name":"critpt-answer"}},"summary":{{"tests":1,"passed":%d,"failed":%d}},"tests":[{{"name":"{ANSWER} parses","status":"%s"}}]}}}}\n' $r $((1-r)) $s > /logs/verifier/ctrf.json
'''


def _cut(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old[:60]
    return text.replace(old, new)


def user_text(p: data.Problem) -> str:
    """prompts.user_prompt with this harness's tools and the answer file in place of the finish tool."""
    t = prompts.user_prompt(p, max_turns=0, cmd_timeout=0)
    t = _cut(t, '- You can run shell commands with the `code_exec` tool, in an offline Linux sandbox.',
             '- You can run shell commands in an offline Linux sandbox.')
    t = _cut(t, '- Each command may run for at most 0 seconds. Files you create stay in the working directory for the '
                'rest of the task.\n',
             f'- Your working directory is `{WORKDIR}`. Files you create stay there for the rest of the task.\n')
    t = _cut(t, '- You have at most 0 turns. Every response you give uses one turn, whether or not it calls a tool.\n',
             '')
    start = t.index('# Submission\n')
    return t[:start] + (
        '# Submission\n'
        f'- Write the complete code template above, with your final answer filled in, to `{ANSWER}`. Keep the '
        'function name and signature, and import nothing beyond what the template already imports.\n'
        '- Give your derivation of the result, step by step, in your final message.\n'
        f'- Only `{ANSWER}` is graded, as it is when you finish. If it is missing or is not valid Python, the '
        'problem scores zero.\n'
        f'- {END_DISCLOSURE}\n')


def pi_instruction(p: data.Problem) -> str:
    return prompts.system_prompt() + '\n' + user_text(p)


def build_task(p: data.Problem, dest: Path) -> dict:
    d = dest / p.problem_id
    (d / 'environment').mkdir(parents=True)
    (d / 'tests').mkdir()
    ins = pi_instruction(p)
    (d / 'instruction.md').write_text(ins)
    (d / 'task.toml').write_text(TASK_TOML.format(answer=ANSWER, version=BUNDLE_VERSION, pid=p.problem_id,
                                                  base=BASE_REF))
    (d / 'environment' / 'Dockerfile').write_text(DOCKERFILE)
    (d / 'tests' / 'test.sh').write_text(TEST_SH)
    (d / 'tests' / 'test.sh').chmod(0o755)
    return {'instruction_sha256': hashlib.sha256(ins.encode()).hexdigest(), 'original_agent_timeout_sec': None}


def _norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ''
    ti.mtime = 0
    return ti


def build(out: Path) -> Path:
    pins = data.check_pins()
    problems = data.load_problems()
    out.mkdir(parents=True, exist_ok=True)
    tar = out / f'{BUNDLE_VERSION}.tar'
    with tempfile.TemporaryDirectory(dir=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') as tmp:
        root = Path(tmp) / 'tasks'
        root.mkdir()
        meta = {p.problem_id: build_task(p, root) for p in problems}
        part = tar.with_suffix('.tar.partial')
        with tarfile.open(part, 'w') as tf:
            for f in sorted(Path(tmp).rglob('*')):
                tf.add(f, arcname=f.relative_to(tmp).as_posix(), recursive=False, filter=_norm)
        part.replace(tar)
    from aa_evals.common.pi_eval.task_inputs import sha256
    manifest = {'bundle_version': BUNDLE_VERSION, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'base_ref': BASE_REF, 'sif_sha256': SIF_SHA256, 'dataset': pins, 'answer_file': ANSWER,
                'system_prompt_sha': prompts.sha(prompts.system_prompt()), 'tasks': meta, 'sha256': sha256(tar)}
    tar.with_suffix('.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{tar}: {len(meta)} tasks, sha256 {manifest["sha256"]}')
    return tar


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['build', 'show'])
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--problem', default='Challenge_1_main')
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        build(Path(a.out))
    else:
        print(pi_instruction({p.problem_id: p for p in data.load_problems()}[a.problem]))


if __name__ == '__main__':
    main()
