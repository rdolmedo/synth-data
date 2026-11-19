"""Per-pipeline config: pipeline.json, found from the calling module (condor_cost.find_pipeline_json): on the trunk
(#480) each pipeline package commits its own (aa_evals/pipeline.json, eval_q35/pipeline.json, synth/ and
task_synthesis/ for task synthesis, task_synthesis/serve/ for the serving launcher), so one tree names every
pipeline; a vendored pipeline branch has one, tbshared/pipeline.json (pipeline-local, sync.LOCAL: never part of the
pinned tree). The nearest file above the caller wins.

  {"pipeline": "aa-evals",                                   # the pipeline's branch name
   "worker_shapes": {                                        # optional; default {}
     "tb21": {"slots": 32, "cpus": 10, "mem_gb": 128,
              "basis": "c30 big-model fit + #378 memrun (one line: where the numbers come from)",
              "cpu_model": {"p95_full_slot": 13.3, "source": "memrun/report.md"}}}}   # cpu_model optional, free-form

  from tbshared import pipeline_config
  pipeline_config.pipeline_name()                 # the caller's pipeline, e.g. 'aa-evals' from aa_evals/... (raises
                                                  # when there is no file or it is malformed)
  pipeline_config.worker_shape('tb21')            # {'slots': 32, ...}; `default` when the file or key is missing
  pipeline_config.load()                          # the whole validated dict
  pipeline_config.find(start=FILE)                # the governing file for FILE (default: the caller's), or None

Validation is strict (unknown keys, wrong types, multi-line `basis` are errors), so a typo fails loudly at load
instead of silently falling back to a default. Stdlib only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / 'pipeline.json'   # a vendored branch's file (none on the trunk)
TOP_KEYS = {'pipeline', 'worker_shapes'}
SHAPE_INTS = ('slots', 'cpus', 'mem_gb')
SHAPE_KEYS = {*SHAPE_INTS, 'basis', 'cpu_model'}


class PipelineConfigError(ValueError):
    """tbshared/pipeline.json is missing or malformed."""


def _validate(cfg, where: str) -> dict:
    def bad(msg):
        raise PipelineConfigError(f'{where}: {msg}')
    if not isinstance(cfg, dict):
        bad(f'expected a JSON object, got {type(cfg).__name__}')
    if set(cfg) - TOP_KEYS:
        bad(f'unknown keys {sorted(set(cfg) - TOP_KEYS)} (allowed: {sorted(TOP_KEYS)})')
    name = cfg.get('pipeline')
    if not isinstance(name, str) or not name.strip():
        bad('"pipeline" must be a non-empty string (the pipeline\'s branch name)')
    shapes = cfg.setdefault('worker_shapes', {})
    if not isinstance(shapes, dict):
        bad('"worker_shapes" must be an object {key: shape}')
    for key, s in shapes.items():
        at = f'worker_shapes[{key!r}]'
        if not isinstance(s, dict):
            bad(f'{at} must be an object')
        if set(s) - SHAPE_KEYS:
            bad(f'{at}: unknown keys {sorted(set(s) - SHAPE_KEYS)} (allowed: {sorted(SHAPE_KEYS)})')
        for k in SHAPE_INTS:
            v = s.get(k)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                bad(f'{at}.{k} must be a positive integer, got {v!r}')
        basis = s.get('basis')
        if not isinstance(basis, str) or not basis.strip() or '\n' in basis:
            bad(f'{at}.basis must be one non-empty line naming the data source')
        if 'cpu_model' in s:
            m = s['cpu_model']
            if not isinstance(m, dict):
                bad(f'{at}.cpu_model must be an object')
            for k, v in m.items():
                if not isinstance(k, str):
                    bad(f'{at}.cpu_model keys must be strings')
                if k == 'source' and not isinstance(v, str):
                    bad(f'{at}.cpu_model.source must be a string')
                if k != 'source' and (isinstance(v, bool) or not isinstance(v, (int, float, str)) or
                                      (isinstance(v, (int, float)) and v < 0)):
                    bad(f'{at}.cpu_model.{k} must be a non-negative number or a string, got {v!r}')
    return cfg


def find(start=None) -> Path | None:
    """The pipeline.json that governs `start` (a file or directory; default: the calling module's file), else the
    one governing tbshared/ itself (a vendored branch's tbshared/pipeline.json; none on the trunk), else None."""
    from tbshared.condor_cost import caller_file, find_pipeline_json
    start = start if start is not None else caller_file()
    f = find_pipeline_json(start) if start is not None else None
    return f if f is not None else find_pipeline_json(Path(__file__).resolve().parent)


def _resolve(path, start):
    if path is not None:
        return Path(path)
    return find(start if start is not None else _caller())


def _caller():
    from tbshared.condor_cost import caller_file
    return caller_file()


def load(path=None, start=None) -> dict:
    """The validated pipeline.json (`path`, else the one governing `start` or the caller). Raises PipelineConfigError
    when it is missing, is not JSON, or does not match the schema."""
    p = _resolve(path, start)
    if p is None:
        raise PipelineConfigError(f'no pipeline.json governs {start or _caller() or "the command line"}: each pipeline '
                                  'package commits its own (tbshared/README.md, "Per-pipeline config")')
    try:
        text = p.read_text()
    except FileNotFoundError:
        raise PipelineConfigError(f'{p}: missing (each pipeline branch commits its own tbshared/pipeline.json; '
                                  'see tbshared/README.md)') from None
    try:
        cfg = json.loads(text)
    except ValueError as e:
        raise PipelineConfigError(f'{p}: not valid JSON ({e})') from None
    return _validate(cfg, str(p))


def worker_shape(key: str, default=None, path=None, start=None):
    """The shape for `key` ({'slots', 'cpus', 'mem_gb', 'basis'[, 'cpu_model']}, a copy), or `default` when the file
    or the key is missing. A malformed file still raises."""
    p = _resolve(path, start)
    if p is None or not p.exists():
        return default
    s = load(p)['worker_shapes'].get(key)
    return copy.deepcopy(s) if s is not None else default


def pipeline_name(path=None, start=None) -> str:
    """The pipeline's name (its branch), e.g. for a numerics record. Raises when the file is missing or malformed."""
    return load(path, start if start is not None or path is not None else _caller())['pipeline']
