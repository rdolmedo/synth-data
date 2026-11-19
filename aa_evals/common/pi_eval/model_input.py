"""Freeze Pi input modalities from endpoint launch evidence, never a model name."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ENV = 'PI_MODEL_INPUT_JSON'


def validate_modalities(value):
    if value not in (['text'], ['text', 'image']) or not isinstance(value, list):
        raise ValueError('Pi model input must be ["text"] or ["text", "image"]')
    return list(value)


def from_env(env):
    if ENV not in env:
        return None
    try:
        return validate_modalities(json.loads(env[ENV]))
    except (ValueError, TypeError) as exc:
        raise ValueError(f'invalid reserved {ENV}: {exc}') from exc


def _image_allowance(argv):
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
        raise ValueError('missing or malformed serving launch argv')
    if '--language-model-only' in argv:
        return False, '--language-model-only'
    allowances = []
    for i, arg in enumerate(argv):
        if arg == '--limit-mm-per-prompt':
            if i + 1 == len(argv):
                raise ValueError('missing --limit-mm-per-prompt value')
            allowances.append(argv[i + 1])
        elif arg.startswith('--limit-mm-per-prompt='):
            allowances.append(arg.split('=', 1)[1])
    if not allowances:
        raise ValueError('no explicit image allowance in serving launch argv')
    if len(allowances) != 1:
        raise ValueError('ambiguous repeated --limit-mm-per-prompt')
    limits = json.loads(allowances[0])
    n = limits.get('image') if isinstance(limits, dict) else None
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('invalid serving image allowance')
    return n > 0, f'--limit-mm-per-prompt image={n}'


def _served_names(argv):
    """Only explicit serving aliases bind launch evidence to an endpoint."""
    if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
        return []
    positions = [i for i, v in enumerate(argv)
                 if v == '--served-model-name' or v.startswith('--served-model-name=')]
    if len(positions) > 1:
        raise ValueError('ambiguous repeated --served-model-name')
    if not positions:
        return []
    i = positions[0]
    names = [argv[i].split('=', 1)[1]] if '=' in argv[i] else []
    for value in argv[i + 1:]:
        if value.startswith('-'):
            break
        names.append(value)
    return [v for v in names if v]


def resolve(ep, declaration='auto'):
    """Return frozen capability/provenance. Unknown AUTO retains the legacy text default.

    Explicit declarations support endpoints without local launch evidence, but
    cannot override a known launch. Conflicting replicas fail before generation.
    """
    if declaration not in ('auto', 'on', 'off'):
        raise ValueError('model images declaration must be auto, on or off')
    capability = None
    source = {'kind': 'unknown', 'why': 'endpoint has no local serving launch evidence'}
    if ep.get('kind') == 'serving':
        path = Path(ep['registry']).parent / 'launch.json'
        source = {'kind': 'unknown', 'path': str(path)}
        try:
            blob = path.read_bytes()
        except OSError as exc:
            source['why'] = f'launch evidence unavailable: {exc}'
        else:
            source['sha256'] = hashlib.sha256(blob).hexdigest()
            source['launch_json_base64'] = base64.b64encode(blob).decode('ascii')
            source['registry_job_id'] = ep.get('job_id')
            try:
                rec = json.loads(blob)
                replicas = rec.get('replicas', [rec])
                if not isinstance(replicas, list) or not replicas:
                    raise ValueError('missing launch replicas')
            except (ValueError, AttributeError, TypeError) as exc:
                source['why'] = f'launch evidence unparsed: {exc}'
            else:
                source['launch_time'] = rec.get('time')
                parsed = []
                for replica in replicas:
                    argv = replica.get('argv') if isinstance(replica, dict) else None
                    names = _served_names(argv)
                    if names and ep.get('served') not in names:
                        raise ValueError('serving launch aliases do not match resolved endpoint served name')
                    try:
                        flag_images, why = _image_allowance(argv)
                    except (ValueError, AttributeError, TypeError) as exc:
                        flag_images, why = None, f'launch evidence unparsed: {exc}'
                    images = flag_images if names else None
                    if not names:
                        why += '; no explicit served-model identity evidence'
                    parsed.append({'images': images, 'launch_flag_images': flag_images,
                                   'served_names': names, 'why': why})
                source['replicas'] = parsed
                known = {p['launch_flag_images'] for p in parsed if p['launch_flag_images'] is not None}
                if len(known) > 1:
                    raise ValueError('serving replicas disagree about Pi image inputs')
                explicit = {'on': True, 'off': False}.get(declaration)
                if explicit is not None and any(v != explicit for v in known):
                    raise ValueError('--model-images contradicts a known serving replica capability')
                if all(p['images'] is not None for p in parsed):
                    capability = parsed[0]['images']
                    source['kind'] = 'serving_launch'
                else:
                    source['why'] = 'one or more serving replicas lack parsed capability evidence'
    explicit = {'on': True, 'off': False}.get(declaration)
    if explicit is not None and capability is not None and explicit != capability:
        raise ValueError('--model-images contradicts the actual serving launch capability')
    images = explicit if explicit is not None else bool(capability)
    return {'modalities': ['text', 'image'] if images else ['text'],
            'resolved_image_capability': capability, 'declaration': declaration, 'source': source,
            'selection': 'explicit declaration' if explicit is not None else
                         'actual serving launch' if capability is not None else 'legacy text-only default (unknown)'}


def frozen_modalities(config):
    record = config.get('model_input')
    protocol = config.get('protocol') or {}
    if record is None:
        if 'model_input' in protocol:
            raise ValueError('Pi protocol input lacks its recorded resolution')
        return None  # historical runs keep their original custom-model defaults
    if not isinstance(record, dict):
        raise ValueError('malformed frozen Pi model input')
    source = record.get('source') or {}
    if 'sha256' in source:
        try:
            snapshot = base64.b64decode(source['launch_json_base64'], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError('missing or malformed frozen Pi launch snapshot') from exc
        if hashlib.sha256(snapshot).hexdigest() != source['sha256']:
            raise ValueError('frozen Pi launch snapshot hash mismatch')
    values = validate_modalities(record.get('modalities'))
    if protocol.get('model_input') != values:
        raise ValueError('Pi config and protocol input modalities disagree')
    return values


def apply_env(config, env):
    """The frozen protocol owns the reserved variable, including request/host overrides."""
    frozen = frozen_modalities(config)
    supplied = from_env(env)
    if supplied is not None and supplied != frozen:
        raise ValueError(f'reserved {ENV} conflicts with frozen Pi model input')
    if frozen is None:
        env.pop(ENV, None)
    else:
        env[ENV] = json.dumps(frozen)
