"""Model specs (aa_evals/README.md): `serving:<name>` (registry, dead if older than 120 s) or `url:<base>@<served>`.

GDPval also takes a comma-separated list of specs serving the same model, each with an optional weight, e.g.
`serving:aa-gdpval-qwen-b200*5,serving:aa-gdpval-qwen*2`: the URLs are pooled (each repeated by its weight) and
an attempt pins to one of them (client.py). A dead entry is skipped while another is alive."""
from __future__ import annotations

import time

from aa_evals.common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)

SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
STALE_SEC = 120


class InfraFailure(Exception):
    """The endpoint could not be reached or kept failing: an infrastructure outcome, never a model one."""


def resolve(spec: str) -> dict:
    """{'served_name', 'urls', 'request'} of a spec or a weighted list of specs (module docstring)."""
    if ',' not in spec and '*' not in spec:
        return _resolve_one(spec)
    eps, errors = [], []
    for part in (p.strip() for p in spec.split(',') if p.strip()):
        one, _, w = part.rpartition('*') if '*' in part else (part, '', '1')
        try:
            eps.append((_resolve_one(one), max(1, int(w))))
        except (InfraFailure, OSError, ValueError) as e:
            errors.append(f'{one}: {e}')
    if not eps:
        raise InfraFailure('; '.join(errors) or f'{spec}: no endpoint')
    names = {ep['served_name'] for ep, _ in eps}
    if len(names) > 1:
        raise ValueError(f'{spec}: the endpoints serve different models {sorted(names)}')
    return {'served_name': eps[0][0]['served_name'], 'request': eps[0][0]['request'],
            'urls': [u for ep, w in eps for u in ep['urls'] for _ in range(w)]}


def _resolve_one(spec: str) -> dict:
    kind, _, rest = spec.partition(':')
    if kind == 'url':
        base, _, served = rest.rpartition('@')
        return {'served_name': served, 'urls': [base.rstrip('/')], 'request': {}}
    if kind != 'serving':
        raise ValueError(f'unknown model spec {spec!r} (serving:<name> | url:<base>@<served_name>)')
    path = serving.path(rest, SERVING_ROOT)
    rec = serving.record(rest, SERVING_ROOT)
    if time.time() - path.stat().st_mtime > STALE_SEC:
        raise InfraFailure(f'{spec}: registry is {time.time() - path.stat().st_mtime:.0f}s old (server dead?)')
    return {'served_name': rec['served_name'], 'urls': [u.rstrip('/') for u in rec['urls']],
            'request': rec.get('request') or {}}


GATE_WAIT_S = 12 * 3600   # total backpressure wait per attempt before the server counts as down


def gate_retry_after(r) -> float | None:
    """Seconds to wait if `r` is an admission gate's backpressure 503 (server_overloaded or X-Gate, with
    Retry-After), else None. Same rule as aa_evals/hle/client.py (#206)."""
    if r is None or getattr(r, 'status_code', None) != 503:
        return None
    try:
        err = (r.json() or {}).get('error') or {}
    except ValueError:
        err = {}
    if not isinstance(err, dict) or (err.get('type') != 'server_overloaded' and r.headers.get('X-Gate') != '1'):
        return None
    try:
        return max(1.0, min(600.0, float(r.headers.get('Retry-After', 60))))
    except ValueError:
        return 60.0
