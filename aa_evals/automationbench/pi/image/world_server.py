"""AutomationBench's simulated SaaS world as a service (#348: AutomationBench under the Pi harness).

  python -I world_server.py --task /task/task.json --state /state [--port 8080]

The image (`world.def`) holds upstream's library at /opt/abworld; this script ships in each task's bundle directory
(`environment/world/`, mounted at /task), so a change to it needs a new bundle, not a new image.

Runs in the task's `world` sidecar (its own sandbox, not the agent's), so the agent changes the world only through
the tools, as under upstream's verifiers environment, and cannot edit the state the grader reads.
- startup: `WorldState(**initial_state)` with `meta.allowed_services` (both precomputed by the bundle builder from
  upstream's `runner.setup_state`: None-stripped initial state, `compute_allowed_services`);
- `POST /call {"tool": NAME, "args": {...}}`: one call of upstream's `api` toolset (api_search, api_fetch,
  base64_encode) on the in-memory world, with upstream's argument handling (`update_tool_args`: an empty-object
  argument means "not given"; `world` injected), the tool's string result, or `str(exception)` as verifiers'
  default error formatter returns; calls are serialized;
- after every call the world is written to `/state/world.json` (atomic) and the call appended to
  `/state/calls.jsonl`; Harbor collects `/state` from this service when the episode ends;
- `/state/hidden.json`: the attributes some api tools keep on the world's service models outside their pydantic
  fields, which `model_dump_json` drops (upstream 4a8e106: `google_sheets._updated_row_keys`, read by the
  `google_sheets_row_(not_)updated` assertions, and `google_ads._offline_jobs`). The grader restores them
  (`pi/run.py restore_hidden`), so it checks the same world object upstream's rubric would;
- `GET /health`.
Imports only the schema and the api tools (pydantic + stdlib): no task builders, assertions or rubric in the image.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import automationbench  # noqa: F401
except ImportError:          # python -I: only the script's directory is on sys.path; the library is in the image
    import sys
    sys.path.insert(0, '/opt/abworld')
from automationbench.schema.world import WorldState  # noqa: E402
from automationbench.tools.api import API_TOOLS  # noqa: E402

TOOLS = {t.__name__: t for t in API_TOOLS}
LOCK = threading.Lock()


def _enc(v):
    """JSON form of a hidden attribute value; sets are tagged so the grader gets a set back."""
    if isinstance(v, (set, frozenset)):
        return {'__set__': sorted((_enc(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))}
    if isinstance(v, dict):
        return {str(k): _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    raise TypeError(f'hidden world attribute value of type {type(v).__name__} is not serializable')


def hidden_state(world: WorldState) -> dict:
    """{service: {attr: value}} for every attribute on the world or one of its service models that is not a
    pydantic field (instance __dict__ extras and pydantic private attributes)."""
    out = {}
    for name, model in [('', world)] + [(f, getattr(world, f)) for f in type(world).model_fields]:
        if not hasattr(model, '__dict__') or not hasattr(type(model), 'model_fields'):
            continue
        extra = {k: v for k, v in vars(model).items() if k not in type(model).model_fields}
        extra.update(getattr(model, '__pydantic_private__', None) or {})
        if extra:
            out[name] = {k: _enc(v) for k, v in extra.items()}
    return out


class World:
    def __init__(self, task: Path, state: Path):
        spec = json.loads(task.read_text())
        self.world = WorldState(**spec['initial_state'])
        self.world.meta.allowed_services = spec.get('allowed_services')
        self.state = state
        self.state.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.save()
        (state / 'calls.jsonl').touch()

    def save(self) -> None:
        for name, text in (('hidden.json', json.dumps(hidden_state(self.world), sort_keys=True)),
                           ('world.json', self.world.model_dump_json())):
            tmp = self.state / f'.{name}.{os.getpid()}'
            tmp.write_text(text)
            os.replace(tmp, self.state / name)

    def call(self, tool: str, args: dict) -> tuple[str, bool]:
        fn = TOOLS.get(tool)
        if fn is None:
            return f'unknown tool {tool!r}; available: {sorted(TOOLS)}', False
        if not isinstance(args, dict):
            return f'Expected tool arguments to be a dict, got {type(args).__name__}: {args}', False
        args = {k: v for k, v in args.items() if not (isinstance(v, dict) and len(v) == 0)}   # update_tool_args
        if 'world' in inspect.signature(fn).parameters:
            args.pop('world', None)
            args['world'] = self.world
        t0 = time.time()
        try:
            out, ok = fn(**args), True
            out = out if isinstance(out, str) else str(out)
        except Exception as e:  # noqa: BLE001  verifiers returns str(e) to the model
            out, ok = f'{e}', False
        self.n += 1
        shown = {k: v for k, v in args.items() if k != 'world'}
        with (self.state / 'calls.jsonl').open('a') as f:
            f.write(json.dumps({'i': self.n, 'tool': tool, 'args': shown, 'ok': ok, 'result_chars': len(out),
                                'result_head': out[:2000], 'sec': round(time.time() - t0, 4),
                                'at': time.time()}, default=str) + '\n')
        self.save()
        return out, ok


def make_server(world: World, host: str = '0.0.0.0', port: int = 8080) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self._send(200, {'ok': True, 'calls': world.n})
            else:
                self._send(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/call':
                return self._send(404, {'error': 'not found'})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                tool, args = req.get('tool'), req.get('args', {})
            except (ValueError, AttributeError) as e:
                return self._send(400, {'error': f'bad request: {e}'})
            with LOCK:
                out, ok = world.call(tool, args)
            self._send(200, {'result': out, 'ok': ok})

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='/task/task.json')
    ap.add_argument('--state', default='/state')
    ap.add_argument('--port', type=int, default=8080)
    a = ap.parse_args()
    make_server(World(Path(a.task), Path(a.state)), port=a.port).serve_forever()


if __name__ == '__main__':
    main()
