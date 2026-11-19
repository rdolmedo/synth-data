"""Model specs of the aa_evals contract (aa_evals/README.md).

  serving:<name>                -> <registry>/<name>/endpoints.json ($TASKSYNTH_SERVING_ROOT, else production)
                                   {served_name, urls: [one per DP rank, http://host:port/v1], updated_at, request}
  url:<base_url>[,<base_url>...]@<served_name>  -> one or more OpenAI-compatible endpoints of one recipe
"""
from __future__ import annotations

import json
import socket
import urllib.request

from aa_evals.common import serving   # the registry reader (tbshared/serving.py, ALIASES copy)

SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
STALE_SEC = 120


def _v1(url: str) -> str:
    url = url.rstrip('/')
    return url if url.endswith('/v1') else url + '/v1'


def resolve(spec: str) -> dict:
    """-> {spec, kind, served, urls, registry?, updated_at?, request?, gate?}"""
    if spec.startswith('serving:'):
        name = spec.split(':', 1)[1]
        path = serving.path(name, SERVING_ROOT)
        d = serving.record(name, SERVING_ROOT)
        return {'spec': spec, 'kind': 'serving', 'name': name, 'registry': str(path), 'served': d['served_name'],
                'urls': [_v1(u) for u in d['urls']], 'updated_at': d.get('updated_at'),
                'request': d.get('request') or {}, 'job_id': d.get('job_id'), 'gate': bool(d.get('gate'))}
    if spec.startswith('url:'):
        base, _, served = spec[4:].rpartition('@')
        if not base or not served:
            raise ValueError(f'model spec {spec!r}: expected url:<base_url>@<served_name>')
        # several engines of one recipe: url:<base1>,<base2>,...@<served> (pi_eval pins each attempt to one of them)
        return {'spec': spec, 'kind': 'url', 'served': served, 'urls': [_v1(b) for b in base.split(',') if b]}
    raise ValueError(f'model spec {spec!r}: expected serving:<name> or url:<base_url>@<served_name>')


def alive(spec: str) -> tuple[bool, dict | None, str]:
    """(usable now, resolved spec, reason). A serving registry older than STALE_SEC means the server is dead."""
    try:
        r = resolve(spec)
    except (OSError, ValueError, KeyError) as e:
        return False, None, f'unresolvable: {e!r}'
    if r['kind'] == 'serving':
        age = serving.age(r)
        if age > STALE_SEC:
            return False, r, f'registry {age:.0f}s old'
        return True, r, 'ok'
    for u in r['urls']:
        try:
            with urllib.request.urlopen(u + '/models', timeout=10) as f:
                if f.status != 200:
                    return False, r, f'bad status from {u}'
        except OSError as e:
            return False, r, f'unreachable {u}: {e!r}'
    return True, r, 'ok'


def ip_url(url: str) -> tuple[str, str]:
    """(url with the host replaced by its IPv4 address, that address). Sandboxes have no cluster DNS search domain."""
    scheme, rest = url.split('://', 1)
    hostport, _, path = rest.partition('/')
    host, _, port = hostport.rpartition(':')
    ip = socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
    return f'{scheme}://{ip}:{port}/{path}'.rstrip('/'), ip


def max_model_len(url: str, served: str) -> int | None:
    try:
        with urllib.request.urlopen(_v1(url) + '/models', timeout=15) as f:
            for m in json.load(f).get('data', []):
                if m.get('id') == served:
                    return m.get('max_model_len')
    except (OSError, ValueError):
        return None
    return None
