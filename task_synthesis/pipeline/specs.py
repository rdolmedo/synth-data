"""Specs for the generator: the asset sampler's draws (asset_sampler/sample_assets.py), a difficulty target, and the
rendering of a spec into the generator's user prompt.

v0 scope: buckets `none` and `best-generated` only (asset sourcing is not built yet). Specs that need a sourced
asset, an upstream project or a VM image are redrawn."""
import json
import random
import subprocess
import sys
from pathlib import Path

from .prompts import ACTIONS, ASSET_USER, GEN_USER, NO_ASSETS, ROLES

SAMPLER = Path(__file__).resolve().parents[1] / 'asset_sampler' / 'sample_assets.py'
BUCKET_MIX = 'none=0.23,best-generated=0.77'  # the blueprint's 15% / 49% anchor shares, renormalized
# Difficulty-target mixes (minutes, weight); round --minutes-mix picks one by name or takes explicit weights.
MIXES = {'easy-heavy': [(15, 0.37), (30, 0.28), (45, 0.14), (60, 0.12), (90, 0.05), (120, 0.03), (180, 0.01)],
         # rounds before #26 (students 1 and 2)
         'student1': [(15, 0.08), (30, 0.17), (45, 0.15), (60, 0.20), (90, 0.15), (120, 0.15), (180, 0.10)]}
DEFAULT_MIX = 'easy-heavy'   # since #26 (AUDIT_LOG.md, 2026-11-02)
MINUTES = MIXES[DEFAULT_MIX]
UNSUPPORTED_CLASSES = {'vm-image', 'upstream-project'}
SCALE = {'snippet': 'a snippet or single small file', 'mini-package': 'a small multi-file package',
         'real-project': 'a realistic project with several modules'}


def supported(spec):
    if spec['bucket'] not in ('none', 'best-generated'):
        return False
    for a in spec.get('assets', []):
        if a['class'] in UNSUPPORTED_CLASSES or a['role'] == 'sourced' or a.get('acquisition'):
            return False
    return True


VOCABS = {'v2': SAMPLER.parent / 'vocab_v2.json',            # #25 designs: v2 (25% new options),
          'v2d2': SAMPLER.parent / 'vocab_v2d2.json',        # D2 flatter, D3 flat (same options, flatter weights)
          'v2d3': SAMPLER.parent / 'vocab_v2d3.json',
          'v2h': SAMPLER.parent / 'vocab_v2h.json',          # hero: v2 with open sub-domain draws 20% -> 5%
          'v2h0': SAMPLER.parent / 'vocab_v2h0.json'}        # v2h, cap_new 0 (TB-fitted top level), TB languages >= 1% (#25)
VOCAB_NAMES = ['v1', *VOCABS]      # v1 = the TB-fitted vocabulary (the sampler without --vocab; students 1-2, #20, #26)
DEFAULT_VOCAB = 'v2h0'             # since 2026-11-04 (#25, AUDIT_LOG.md; v2h 11-03 to 11-04); --vocab v1 draws as before
_vocab_cache = {}


def vocab(name):
    """A broadened spec vocabulary (#25) by name, or None for the TB-fitted v1."""
    if not name or name == 'v1':
        return None
    if name not in _vocab_cache:
        _vocab_cache[name] = json.loads(VOCABS[name].read_text())
    return _vocab_cache[name]


def parse_mix(text):
    """A MIXES name, or explicit weights 'minutes=weight,...' (e.g. '15=1,30=1'; weights need not sum to 1)."""
    if text in MIXES:
        return MIXES[text]
    try:
        mix = [(int(m), float(w)) for m, w in (x.split('=') for x in text.split(','))]
    except ValueError:
        mix = []
    if not mix or any(m <= 0 or w < 0 for m, w in mix) or not sum(w for _, w in mix) > 0:
        raise ValueError(f'minutes mix {text!r}: one of {sorted(MIXES)} or minutes=weight,... with positive total weight')
    return mix


def sample(n, seed, minutes=None, mix=None, vocab_name=None):
    """n supported specs, each with a difficulty target (drawn from `mix`, default MINUTES, or `minutes` for all);
    deterministic in seed. vocab_name: one of VOCAB_NAMES ('v1' = the TB-fitted vocabulary), default DEFAULT_VOCAB."""
    mix = mix or MINUTES
    vocab_name = vocab_name or DEFAULT_VOCAB
    out, draw = [], 0
    rng = random.Random(seed)
    extra = [] if vocab_name == 'v1' else ['--vocab', str(VOCABS[vocab_name])]
    while len(out) < n:
        raw = subprocess.run([sys.executable, '-I', str(SAMPLER), '--n', str(4 * n), '--seed', str(seed * 1000 + draw),
                              '--bucket-mix', BUCKET_MIX, *extra], capture_output=True, text=True, check=True).stdout
        draw += 1
        for line in raw.splitlines():
            spec = json.loads(line)
            if supported(spec):
                drawn = rng.choices([m for m, _ in mix], [w for _, w in mix])[0]
                spec['minutes'] = minutes or drawn
                spec['spec_id'] = f's{seed}-{len(out):05d}'
                out.append(spec)
                if len(out) == n:
                    break
    return out


CLASS_NAMES = {'python-code': 'Python code', 'c-cpp-code': 'C/C++ code', 'other-code': 'source code',
               'git-repo': 'git repository', 'tabular-data': 'tabular data', 'text-data': 'text data',
               'structured-data': 'structured data', 'running-service': 'running service',
               'ml-model': 'machine-learning model', 'binary': 'binary or executable', 'database': 'database',
               'media': 'media file', 'document': 'document', 'config': 'configuration', 'archive': 'archive',
               'scientific-data': 'scientific data'}


def describe_asset(a, labels=None, no_toolchain=()):
    labels = labels or {}
    bits = [CLASS_NAMES.get(a['class'], a['class'].replace('-', ' '))]
    detail = [labels.get(f, f) for f in list(a.get('formats', [])) + list(a.get('services', []))]
    if a.get('language'):
        detail.insert(len(a.get('formats', [])), a['language'])
    if detail:
        bits.append(f'({", ".join(detail)})')
    role = ROLES.get(a['role'], a['role'])
    if a.get('state'):
        state = a['state'] if isinstance(a['state'], list) else [a['state']]
        role += f'; state: {", ".join(labels.get(str(x), str(x)) for x in state)}'
    if a.get('language') == 'CUDA':
        role += '; no GPU here, so the CUDA code can be read, ported or analyzed but not run'
    elif a.get('language') in no_toolchain:
        role += f'; no {a["language"]} toolchain installs offline here, so the code can be read, ported or analyzed but not run'
    return f'  - {" ".join(bits)}: {role}'


def render(spec, asset_render=None, agentic=False, asset_first=False, commands=0, workspace=False, tokens=0, lean=False):
    """asset_render: for a spec bound to a real asset (spec['asset']), the asset's files as text
    (task_synthesis.assets.render.for_generator). agentic: the generator explores the asset in a sandbox first
    (pipeline.explore); asset_first: the generator also chooses the action (the spec keeps domain and minutes)."""
    from .prompts import ASSET_AGENTIC_INTRO, ASSET_EXPLORE, ASSET_FIRST_USER, ASSET_LEAN, ASSET_LEAN2, ASSET_LEAN3_EXTRA, ASSET_ONESHOT_INTRO, ASSET_WORKSPACE
    a = spec.get('asset')
    intro = ASSET_AGENTIC_INTRO if agentic else ASSET_ONESHOT_INTRO
    asset_block = ''
    if a:
        asset_block = ASSET_USER.format(intro=intro, title=a.get('title'), source=a.get('source'), license=a.get('license'),
                                        description=(a.get('description') or '').strip()[:600], idea=a.get('idea') or '-',
                                        render=asset_render or '(not rendered)')
        if workspace:
            asset_block += '\n' + ASSET_WORKSPACE.format(commands=commands, minutes=spec['minutes'], tokens=tokens // 1000)
            if lean:
                asset_block += '\n' + {'lean2': ASSET_LEAN2, 'lean3': ASSET_LEAN2 + '\n' + ASSET_LEAN3_EXTRA}.get(lean, ASSET_LEAN) + '\n'
        elif agentic:
            asset_block += '\n' + ASSET_EXPLORE.format(commands=commands)
    if a and asset_first:
        return ASSET_FIRST_USER.format(domain=spec['domain'].replace('-', ' '), minutes=spec['minutes'],
                                       actions='; '.join(ACTIONS.values())).rstrip() + '\n' + asset_block
    v = vocab(spec.get('vocab'))
    labels = v['labels'] if v else {}
    new_actions = {x['name']: x['description'] for x in v['actions']} if v else {}
    action = ACTIONS.get(spec.get('agent_action')) or new_actions.get(spec.get('agent_action'), spec.get('agent_action'))
    if not spec.get('has_assets'):
        materials, extra = NO_ASSETS + '\n', ''
    else:
        lines = [describe_asset(a, labels, v['no_toolchain'] if v else ()) for a in spec.get('assets', [])]
        if spec.get('asset'):
            lines[0] += '; a REAL, public asset (see below)'
        materials = '- Materials in the environment:\n' + '\n'.join(lines) + '\n'
        if spec.get('agent_works_on'):
            works = spec['agent_works_on']
            materials += f'- The agent mainly works on: {CLASS_NAMES.get(works, works.replace("-", " "))}\n'
        if spec.get('code_scale') in SCALE:
            materials += f'- Size of the code involved: {SCALE[spec["code_scale"]]}\n'
        extra = ''
        if spec.get('shipped_size'):
            extra = f'- Total size of the shipped materials: {spec["shipped_size"]}\n'
    domain = spec['domain'].replace('-', ' ') + (f'; area: {spec["subdomain"]}' if spec.get('subdomain') else '')
    text = GEN_USER.format(domain=domain, action=action, materials=materials,
                           minutes=spec['minutes'], extra=extra).rstrip() + '\n'
    return text + asset_block


def spec_text(spec):
    """Compact spec for the judge."""
    keep = ('domain', 'subdomain', 'agent_action', 'agent_works_on', 'code_scale', 'minutes')
    return json.dumps({**{k: spec[k] for k in keep if spec.get(k)}, 'assets': spec.get('assets', [])})
