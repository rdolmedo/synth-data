"""AutomationBench tasks as Harbor tasks for the TB 2.1 Pi harness (#348).

  python -m aa_evals.automationbench.pi.tasks build      # once, in the AutomationBench venv (re-executes itself)
  python -m aa_evals.automationbench.pi.tasks build --tasks finance.cash_sweep --oracle --out DIR --version V
                                                         # a plumbing bundle: solution/solve.sh drives `ab`
  python -m aa_evals.automationbench.pi.tasks show [--task sales.x]

Upstream runs each task as a verifiers StatefulToolEnv: the model calls the `api` toolset (api_search, api_fetch,
base64_encode) as native tool calls on an in-process simulated world, and the final world is checked with the
task's assertions. Here each of the 600 public tasks becomes a Harbor task that `aa_evals.common.pi_eval` runs like
a TB 2.1 task, with the same world and tools behind a command:
- `world` service (compose sidecar, its own sandbox): `image/world_server.py` on upstream's schema and api tools
  (shipped in the task's `environment/world/`; image `WORLD_REF`, `image/world.def`: upstream's library), started from the task's initial world (`environment/world/task.json`:
  upstream's None-stripped `initial_state` and `compute_allowed_services`, as `runner.setup_state` builds them).
  The world lives only there, so the agent can change it only through the tools, and it persists for the episode.
  Harbor collects its `/state` (world.json after every call, calls.jsonl) when the episode ends.
- `main` (the agent's sandbox, `AGENT_REF`: python:3.13-slim): `ab <tool> '<json args>'` (`image/ab`) posts one
  tool call to the world service and prints the tool's result; the tool list (upstream's own tool schemas) is in
  the instruction and in `ab tools`.
- instruction: the task's upstream system prompt (without its "~50 tool-using turns" budget sentence: this
  protocol has no turn cap), how to call the tools, and the task's request (the upstream user message), verbatim.
- grading stays in `aa_evals/automationbench` (`pi/run.py`): the collected final world against the task's
  assertions with AA's scoring (`scoring.py`), as for upstream rollouts. `tests/test.sh` only checks the world
  service still answers (reward 1/0), not the score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORLD_REF = 'aa-evals.local/automationbench-world:v1'
AGENT_REF = 'aa-evals.local/automationbench-agent:v1'
BUNDLE_VERSION = 'automationbench-pi-v2'   # v1: the first instruction wording (smoke only)
OUT = Path('/data/aai/aa_evals/data/automationbench/pi_bundle')
DOMAINS = ['sales', 'marketing', 'operations', 'support', 'finance', 'hr']
BUDGET = 'You have a budget of ~50 tool-using turns — favor parallel tool calls and avoid duplicate searches.'
BUDGET_NEW = 'Avoid duplicate searches.'

TASK_TOML = '''schema_version = "1.1"
artifacts = [{{ source = "/state", service = "world" }}]

[metadata]
benchmark = "automationbench"
bundle_version = "{version}"
task_name = "{task}"
example_id = {example_id}

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 21600.0

[environment]
build_timeout_sec = 1800.0
docker_image = "{agent}"
cpus = 1
memory_mb = 4096
storage_mb = 4096
gpus = 0
allow_internet = false
'''

COMPOSE = f'''services:
  main:
    depends_on:
      world:
        condition: service_healthy
  world:
    image: {WORLD_REF}
    command: ["python", "-I", "/task/world_server.py", "--task", "/task/task.json", "--state", "/state"]
    environment:
      PYTHONPATH: /opt/abworld
    volumes:
      - ./world:/task:ro
    healthcheck:
      test: ["CMD", "python", "-I", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]
      interval: 1s
      timeout: 5s
      retries: 120
'''

TEST_SH = '''#!/bin/bash
# The world service still answers (its /state is collected as an artifact); the score comes from the assertions.
mkdir -p /logs/verifier
if python3 -I -c "import urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://world:8080/health', timeout=10)"; then
  r=1; s=passed; echo "world service answers"
else
  r=0; s=failed; echo "world service does not answer"
fi
echo $r > /logs/verifier/reward.txt
# a CTRF report, so pi_eval's verifier wrapper (eval_q35 VERIFIER_WRAPPER) sees a finished check and does not retry
printf '{"results":{"tool":{"name":"automationbench-world"},"summary":{"tests":1,"passed":%d,"failed":%d},"tests":[{"name":"world service answers","status":"%s"}]}}\\n' $r $((1-r)) $s > /logs/verifier/ctrf.json
'''


def ensure_venv(mod: str = 'aa_evals.automationbench.pi.tasks'):
    try:
        import automationbench  # noqa: F401
        import verifiers  # noqa: F401
    except ImportError:
        from aa_evals.automationbench.run import VENV
        py = os.path.join(VENV, 'bin', 'python')
        if os.environ.get('_AA_AB_REEXEC'):
            raise SystemExit(f'automationbench is not importable from {sys.executable}')
        root = str(HERE.parents[2])
        env = dict(os.environ, _AA_AB_REEXEC='1', PYTHONNOUSERSITE='1',
                   PYTHONPATH=root + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
        os.execve(py, [py, '-m', mod, *sys.argv[1:]], env)


def tool_schemas() -> list[dict]:
    """Upstream's api toolset in OpenAI function format, exactly what the verifiers path sends as `tools`."""
    from automationbench.rubric import create_rubric
    from automationbench.runner import AutomationBenchEnv
    from datasets import Dataset
    rows = load_rows()
    env = AutomationBenchEnv(dataset=Dataset.from_list([{k: v for k, v in rows[0].items() if not k.startswith('_')}]),
                             rubric=create_rubric(), max_turns=50, toolset='api')
    return list(env._all_oai_tools)


def load_rows() -> list[dict]:
    from aa_evals.automationbench.run import load_tasks
    return load_tasks(DOMAINS)


def tools_text(tools: list[dict]) -> str:
    """Each tool's name, description and argument schema (what upstream sends as `tools`), as `ab` subcommands."""
    out = []
    for t in tools:
        f = t['function']
        out.append(f'### ab {f["name"]}\n\n{(f.get("description") or "").strip()}\n\n'
                   f'JSON arguments (schema): {json.dumps(f.get("parameters") or {})}\n')
    return '\n'.join(out)


# How to reach the tools: the only text of the instruction that is not upstream's. v1 (smoke, 2026-11-11): Qwen3.5-4B
# called `ab` as a native tool and searched the sandbox's filesystem for the task's "files" (they are in the apps).
TOOLS_HOWTO = '''## How to use the workspace's apps

Everything this task refers to (spreadsheets, documents, emails, messages, records) lives in the workspace's apps.
You reach them only through the `ab` command, which you run with your bash tool; the local filesystem has nothing
for this task. Run one tool per command:

    ab <tool> '<JSON arguments>'

where `<tool>` is one of {names} (described below) and the JSON object holds only that tool's arguments. Examples:

    ab api_search '{{"query": "update a salesforce opportunity"}}'
    ab api_fetch '{{"method": "GET", "url": "<url from api_search>"}}'

For arguments that are hard to quote, pass `-` and give the JSON on stdin:

    ab api_fetch - <<'EOF'
    {{"method": "POST", "url": "<url from api_search>", "body": "<the request body as a JSON string>"}}
    EOF

`ab` prints the tool's result. Changes you make persist for the whole task. `ab tools` prints the tool list as JSON.

## The tools (`ab` subcommands)

'''


def instruction(row: dict, tools: list[dict]) -> str:
    """Upstream's system prompt (its turn-budget sentence replaced: no turn cap here), how to reach the tools
    (TOOLS_HOWTO + the tool schemas), then upstream's request message verbatim."""
    msgs = row['prompt']
    system = [m['content'] for m in msgs if m['role'] == 'system']
    user = [m['content'] for m in msgs if m['role'] == 'user']
    assert len(system) == 1 and len(user) == 1, row['_task_name']
    s = system[0]
    assert s.count(BUDGET) == 1, row['_task_name']
    s = s.replace(BUDGET, BUDGET_NEW)
    names = ', '.join(t['function']['name'] for t in tools)
    return (s + '\n\n' + TOOLS_HOWTO.format(names=names) + tools_text(tools) + '\n## Request\n\n' + user[0])


# `--oracle`: Harbor's oracle agent runs this in the agent's sandbox. It is no solution of the task: it checks the
# plumbing (the `ab` CLI, the world service from the agent box, state that persists across calls, collection).
ORACLE_SH = r'''#!/bin/bash
set -u
ab tools > /tmp/tools.json && python3 -c "import json; print('tools', [t['function']['name'] for t in json.load(open('/tmp/tools.json'))])"
ab api_search '{"query": "slack post message channel", "top_k": 2}' | head -c 400; echo
cat <<'JSON' | ab api_fetch -
{"method": "POST", "url": "https://slack.com/api/chat.postMessage", "params": "{}", "body": "{\"channel\": \"#treasury\", \"text\": \"pi-oracle-plumbing\"}"}
JSON
ab api_fetch '{"method": "GET", "url": "https://slack.com/api/conversations.history", "params": "{\"channel\": \"#treasury\"}"}' | grep -c pi-oracle-plumbing
ab no_such_tool '{}'; echo "exit $?"
ab api_fetch 'not json'; echo "exit $?"
'''


def world_spec(row: dict) -> dict:
    """The world service's start state, as upstream's `runner.setup_state` builds it: the None-stripped initial
    state and `compute_allowed_services` over it, the task's assertions and its Zapier tools."""
    from automationbench.runner import compute_allowed_services, strip_none_values
    info = json.loads(row['info']) if isinstance(row['info'], str) else row['info']
    initial = strip_none_values(info.get('initial_state', {}))
    allowed = compute_allowed_services(initial, [strip_none_values(a) for a in info.get('assertions', [])],
                                       info.get('zapier_tools', []))
    return {'task_name': row['_task_name'], 'example_id': row['example_id'], 'initial_state': initial,
            'allowed_services': allowed}


def build_task(row: dict, tools: list[dict], dest: Path, oracle: bool = False) -> dict:
    from aa_evals.common.pi_eval import task_inputs
    spec = world_spec(row)
    allowed = spec['allowed_services']
    d = dest / row['_task_name']
    for sub in ('environment/world', 'tests', 'inputs/usr/local/bin', 'inputs/opt/ab'):
        (d / sub).mkdir(parents=True)
    (d / 'environment/world/task.json').write_text(json.dumps(spec))
    (d / 'environment/docker-compose.yaml').write_text(COMPOSE)
    shutil.copy2(HERE / 'image' / 'world_server.py', d / 'environment/world/world_server.py')
    shutil.copy2(HERE / 'image' / 'ab', d / 'inputs/usr/local/bin/ab')
    (d / 'inputs/usr/local/bin/ab').chmod(0o755)
    (d / 'inputs/opt/ab/tools.json').write_text(json.dumps(tools, indent=1))
    ins = instruction(row, tools)
    (d / 'instruction.md').write_text(ins)
    (d / 'task.toml').write_text(TASK_TOML.format(version=BUNDLE_VERSION, task=row['_task_name'],
                                                  example_id=json.dumps(row['example_id']), agent=AGENT_REF))
    if oracle:
        (d / 'solution').mkdir()
        (d / 'solution/solve.sh').write_text(ORACLE_SH)
        (d / 'solution/solve.sh').chmod(0o755)
    (d / 'tests/test.sh').write_text(TEST_SH)
    (d / 'tests/test.sh').chmod(0o755)
    task_inputs.write_manifest(d)
    return {'instruction_sha256': hashlib.sha256(ins.encode()).hexdigest(), 'example_id': row['example_id'],
            'allowed_services': allowed, 'original_agent_timeout_sec': None}


def _norm(ti: tarfile.TarInfo) -> tarfile.TarInfo:
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ''
    ti.mtime = 0
    return ti


def build(out: Path, version: str = BUNDLE_VERSION, only: list[str] | None = None, oracle: bool = False) -> Path:
    from aa_evals.automationbench.run import contract_digest
    from aa_evals.common.pi_eval.task_inputs import sha256
    rows = load_rows()
    tools = tool_schemas()
    if only:
        rows = [r for r in rows if r['_task_name'] in set(only)]
        assert len(rows) == len(set(only)), 'unknown --tasks'
    out.mkdir(parents=True, exist_ok=True)
    tar = out / f'{version}.tar'
    with tempfile.TemporaryDirectory(dir=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') as tmp:
        root = Path(tmp) / 'tasks'
        root.mkdir()
        meta = {r['_task_name']: build_task(r, tools, root, oracle) for r in rows}
        part = tar.with_suffix('.tar.partial')
        with tarfile.open(part, 'w') as tf:
            for f in sorted(Path(tmp).rglob('*')):
                tf.add(f, arcname=f.relative_to(tmp).as_posix(), recursive=False, filter=_norm)
        part.replace(tar)
    from aa_evals.automationbench.run import UPSTREAM_COMMIT, UPSTREAM_VERSION
    bases = Path('/data/aai/tbench/harbor_ns/bases')
    images = {}
    for ref in (WORLD_REF, AGENT_REF):
        m = bases / (re.sub(r'[^A-Za-z0-9_.-]+', '_', ref) + '.json')   # harbor_ns.builder._safe_name
        images[ref] = json.loads(m.read_text()).get('source') if m.exists() else None
    manifest = {'bundle_version': version, 'oracle_plumbing': oracle, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'upstream': {'commit': UPSTREAM_COMMIT, 'version': UPSTREAM_VERSION},
                'task_contract_sha256': contract_digest(rows), 'domains': DOMAINS, 'images': images,
                'scripts': {n: sha256(HERE / 'image' / n) for n in ('world_server.py', 'ab')},
                'tools': tools, 'tasks': meta, 'sha256': sha256(tar)}
    tar.with_suffix('.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'{tar}: {len(meta)} tasks, sha256 {manifest["sha256"]}')
    return tar


def main(argv=None):
    ensure_venv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['build', 'show'])
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--task', default=None)
    ap.add_argument('--tasks', default='', help='build: only these tasks (comma-separated)')
    ap.add_argument('--oracle', action='store_true', help='build: add the plumbing solution/solve.sh')
    ap.add_argument('--version', default=BUNDLE_VERSION)
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        if (a.oracle or a.tasks) and (a.version == BUNDLE_VERSION or Path(a.out) == OUT):
            ap.error('a partial or oracle bundle needs its own --out and --version')
        build(Path(a.out), a.version, [t for t in a.tasks.split(',') if t] or None, a.oracle)
    else:
        rows = {r['_task_name']: r for r in load_rows()}
        print(instruction(rows[a.task or sorted(rows)[0]], tool_schemas()))


if __name__ == '__main__':
    main()
