"""Specs for the TB-Sci-lite generator (#353): draws from vocab.json and the rendering of a spec into the user
prompt. A spec keeps M1's keys where the round reads them (spec_id, domain = the field, agent_action = the
archetype, minutes, asset) so task_synthesis.pipeline.round runs it unchanged."""
import json
import random
from collections import Counter
from pathlib import Path

from . import prompts as P

VOCAB_PATH = Path(__file__).resolve().parent / 'vocab.json'
_vocab = None


def vocab():
    global _vocab
    if _vocab is None:
        _vocab = json.loads(VOCAB_PATH.read_text())
    return _vocab


def _pick(rng, weights):
    keys = list(weights)
    return rng.choices(keys, [weights[k] for k in keys])[0]


def field_modalities(v, field):
    """Modalities the field's brainstormed formats cover (model-spec always: equations, netlists, instances)."""
    return {x['modality'] for x in v['formats'].get(field, [])} | {'model-spec'}


# The pilot's 2-6 h target mix (vocab v1). vocab.json v2 carries 4-12 h; the coordinator gates v2's targets on the
# v2a round's Super submit share (>= 60%), so production can draw either.
MINUTES_V1 = [[120, 0.35], [180, 0.30], [240, 0.20], [360, 0.15]]


def sample(n, seed, datasets=None, real_share=None, start=0, minutes=None, prior_uses=None, ms_boost=1.0):
    """n specs, deterministic in seed. datasets: {dataset_id: record} of the shared scidata cache (scidata.py);
    a spec goes to real data with probability real_share when a dataset fits its field (and modality, if one
    does), the least used dataset of the field first (prior_uses: {dataset_id: uses in earlier rounds}).
    minutes: [[minutes, weight], ...] in place of the vocabulary's target mix.
    ms_boost: P(model-spec | archetype) is multiplied by this (capped at 0.95, the other modalities rescaled), keeping
    the archetype mix; an author turns about a third of planned model-spec specs into a dataset task (prod2 labels)."""
    v = vocab()
    rng = random.Random(f'tbsci-{seed}' + (f'-from{start}' if start else ''))   # an appended batch never repeats draws
    real_share = v['real_share'] if real_share is None else real_share
    uses = Counter(prior_uses or {})
    minutes_mix = minutes or v['minutes']
    joint = v.get('modality_by_archetype')   # vocab v3: the main data follows the kind of work, as in lite
    if joint and ms_boost != 1.0:
        def boost(w):
            ms = min(0.95, w.get('model-spec', 0.0) * ms_boost)
            rest = sum(p for m, p in w.items() if m != 'model-spec')
            return {m: (ms if m == 'model-spec' else p * (1 - ms) / rest if rest else 0.0) for m, p in w.items()}
        joint = {k: boost(w) for k, w in joint.items()}
    if joint:
        # real data goes to data-driven kinds of work, keeping the overall real share: p_real(arch) is real_share
        # scaled by the arch's non-model-spec mass over its mean under the pooled archetype mix
        ap = {k: w for k, w in v['archetype_pooled'].items() if k != 'prove-derive'}
        mean_data = sum(w * (1 - joint[k]['model-spec']) for k, w in ap.items()) / sum(ap.values())
        p_real = {k: min(1.0, real_share * (1 - joint[k]['model-spec']) / mean_data) for k in joint}
    else:
        p_real = {k: real_share for k in v['archetypes']}
    by_field = {}
    for d in (datasets or {}).values():
        by_field.setdefault(d['field'], []).append(d)
    out = []
    for i in range(start, start + n):
        dom = _pick(rng, {d: x['weight'] for d, x in v['domains'].items()})
        field = _pick(rng, v['domains'][dom]['fields'])
        subs = v['subtopics'][field]
        sub = None if rng.random() < v['open_subtopic'] else rng.choice(subs)
        arch = _pick(rng, v['archetype_by_domain'][dom])
        source = 'simulated'
        dataset = None
        if arch != 'prove-derive' and by_field.get(field) and rng.random() < p_real[arch]:
            source = 'real'
        if source == 'real':
            pool = by_field[field]
            low = min(uses[d['id']] for d in pool)
            dataset = rng.choice([d for d in pool if uses[d['id']] == low])
            uses[dataset['id']] += 1
            modality = dataset.get('modality') or 'tabular'
            fmt = dataset.get('format')
        else:
            if arch == 'prove-derive':
                modality = 'model-spec'
            else:
                allowed = field_modalities(v, field)
                w = {m: p for m, p in (joint[arch] if joint else v['modality_by_domain'][dom]).items() if m in allowed}
                modality = _pick(rng, w)
            fmts = [x for x in v['formats'].get(field, []) if x['modality'] == modality]
            fmt = None
            if fmts and rng.random() < 0.7:   # 30%: the author picks the format
                fmt = rng.choices(fmts, [2 if x.get('share') == 'common' else 1 for x in fmts])[0]['format']
        minutes = rng.choices([m for m, _ in minutes_mix], [w for _, w in minutes_mix])[0]
        closing = rng.choices([s for s, _ in P.CLOSING_SECONDS], [w for _, w in P.CLOSING_SECONDS])[0]
        spec = {'spec_id': f's{seed}-{i:05d}', 'bench': 'tbsci_lite', 'sci_domain': dom, 'domain': field,
                'subtopic': sub, 'agent_action': arch, 'modality': modality, 'format': fmt, 'source': source,
                'minutes': minutes, 'closing_seconds': closing, 'vocab_version': v['version']}
        if dataset:
            spec['asset'] = {'dir': dataset['dir'], 'id': dataset['id'], 'title': dataset['title'],
                             'source': dataset['source'], 'license': dataset['license'], 'url': dataset['url'],
                             'sha256': dataset['sha256'], 'description': dataset.get('description', '')}
        out.append(spec)
    return out


def render(spec, asset_render=None, **_):
    """The generator's user prompt (replaces task_synthesis.pipeline.specs.render; the one-shot author's options are
    ignored)."""
    v = vocab()
    field = v['field_desc'].get(spec['domain'], spec['domain'])
    sub = spec.get('subtopic')
    subtopic = f'{sub["name"]} ({sub["gist"]})' if sub else P.OPEN_SUBTOPIC
    fmt = f'; format: {spec["format"]}' if spec.get('format') else ''
    text = P.GEN_USER.format(field=field, subtopic=subtopic,
                             archetype=v['archetypes'][spec['agent_action']],
                             modality=v['modalities'][spec['modality']], fmt=fmt,
                             source=P.SOURCE_REAL if spec.get('asset') else P.SOURCE_SIM, minutes=spec['minutes'])
    a = spec.get('asset')
    if a:
        text += P.ASSET_USER.format(intro=P.ASSET_INTRO, title=a.get('title'), source=a.get('source'),
                                    license=a.get('license'), description=(a.get('description') or '').strip()[:600],
                                    render=asset_render or '(not rendered)')
    return text


def spec_text(spec):
    """Compact spec for the judge."""
    keep = ('domain', 'subtopic', 'agent_action', 'modality', 'format', 'source', 'minutes')
    return json.dumps({k: spec[k] for k in keep if spec.get(k)})
