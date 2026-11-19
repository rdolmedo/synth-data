"""Hugging Face Hub: datasets (tabular, structured and text files) and small models (weights in one format, configs,
tokenizers). Everything pins to the repository's commit sha (resolve/<sha>/<path>). Gated and private repositories
are skipped; synthetic datasets too (we want real data: ids, authors or card text saying 'synthetic' or 'synth-').
Licenses come from the card ('license:' tags or cardData.license); 'other' and 'unknown' are excluded by the policy.
"""
import gzip
import hashlib
import re
from pathlib import Path

from ..http import TooLarge
from .base import Reject, Source, cap

HUB = 'https://huggingface.co'
SYNTHETIC = re.compile(r'synthetic|\bsynth\b|synth-|-synth|infinite-dataset-hub|'
                       r'(generated|written|created|produced) (by|with|using) (an? )?(gpt|chatgpt|llms?|language models?|'
                       r'claude|gemini|ai)\b', re.I)
LLM_ID = re.compile(r'llm|gpt|chatgpt|claude|gemini|distil(?:led)?-from', re.I)  # in the dataset id: model-written
BLOCKED_AUTHORS = {'infinite-dataset-hub'}

DATA_EXT = {'csv': ['.csv'], 'tsv': ['.tsv'], 'parquet': ['.parquet'], 'json': ['.json'], 'jsonl': ['.jsonl', '.ndjson'],
            'plain-text': ['.txt'], 'excel': ['.xlsx'], 'feather': ['.feather', '.arrow'], 'xml': ['.xml'],
            'log': ['.log', '.txt'], 'yaml': ['.yaml', '.yml']}
DATA_DEFAULT = {'tabular-data': ['csv', 'parquet', 'tsv'], 'structured-data': ['jsonl', 'json'],
                'text-data': ['plain-text']}
FORMAT_TAG = {'csv': 'format:csv', 'tsv': 'format:csv', 'parquet': 'format:parquet', 'json': 'format:json',
              'jsonl': 'format:json', 'plain-text': 'format:text'}
MODALITY = {'tabular-data': 'modality:tabular', 'text-data': 'modality:text'}
SKIP_FILES = ('.gitattributes', 'README.md', 'readme.md')


def card_license(x):
    tags = [t[len('license:'):] for t in x.get('tags', []) if t.startswith('license:')]
    card = (x.get('cardData') or {}).get('license')
    if isinstance(card, list):
        card = card[0] if card else None
    return card or (tags[0] if tags else None)


def file_ext(path, exts):
    low = path.lower()
    if low.endswith('.gz'):
        low = low[:-3]
    return next((e for e in exts if low.endswith(e)), None)


def pick(files, exts, limit, k=10, min_bytes=1000):
    """Matching files (size known), data/ first, up to k within limit."""
    match = [f for f in files if f.get('type') == 'file' and file_ext(f['path'], exts)
             and Path(f['path']).name not in SKIP_FILES and f.get('size', 0) >= min_bytes]
    match.sort(key=lambda f: (0 if '/' in f['path'] else 1, f['path']))
    out, total = [], 0
    for f in match:
        if total + f['size'] > limit:
            continue
        out.append(f)
        total += f['size']
        if len(out) == k:
            break
    return out, total


async def tree(http, kind, repo, rev):
    return await http.json(f'{HUB}/api/{kind}/{repo}/tree/{rev}', params={'recursive': 'true'})


def gunzip(path, limit):
    out = path.with_name(path.name[:-3])
    n = 0
    with gzip.open(path, 'rb') as src, open(out, 'wb') as dst:
        while chunk := src.read(1 << 20):
            n += len(chunk)
            if n > limit:
                raise Reject(f'{path.name} decompresses past {limit} bytes')
            dst.write(chunk)
    path.unlink()


class HfDatasets(Source):
    name = 'hf-dataset'
    classes = ('tabular-data', 'structured-data', 'text-data')

    def formats(self, target):
        fmts = [f for f in (target.get('formats') or []) if f in DATA_EXT] or DATA_DEFAULT.get(target['cls'], ['csv'])
        return fmts, sorted({e for f in fmts for e in DATA_EXT[f]})

    async def search(self, target, query, n):
        fmts, _ = self.formats(target)
        params = [('search', query), ('limit', str(max(20, n * 4))), ('full', 'true'), ('sort', 'downloads'),
                  ('direction', '-1')]
        if target['cls'] in MODALITY:
            params.append(('filter', MODALITY[target['cls']]))
        tag = FORMAT_TAG.get(fmts[0]) if len(fmts) == 1 else None
        if tag:
            params.append(('filter', tag))
        items = await self.http.json(f'{HUB}/api/datasets', params=params)
        out = []
        for x in items:
            text = f'{x["id"]} {(x.get("description") or "")[:600]}'
            if x.get('gated') or x.get('private') or x.get('disabled') or x.get('author') in BLOCKED_AUTHORS \
                    or SYNTHETIC.search(text) or LLM_ID.search(x['id'].split('/')[-1]):
                continue
            out.append({'source': self.name, 'key': x['id'], 'url': f'{HUB}/datasets/{x["id"]}',
                        'page_url': f'{HUB}/datasets/{x["id"]}', 'title': x['id'],
                        'description': re.sub(r'\s+', ' ', x.get('description') or '').strip()[:400],
                        'license_raw': card_license(x), 'size': None,
                        'formats': [t[len('format:'):] for t in x.get('tags', []) if t.startswith('format:')],
                        'meta': {'sha': x.get('sha'), 'downloads': x.get('downloads'), 'likes': x.get('likes'),
                                 'tags': [t for t in x.get('tags', []) if t.startswith(('size_categories:',
                                                                                         'task_categories:'))][:6]},
                        'query': query})
        return out[: n * 3]

    async def fetch(self, cand, staging, target):
        staging, limit = Path(staging), cap(target)
        _, exts = self.formats(target)
        sha = cand['meta'].get('sha') or 'main'
        files, total = pick(await tree(self.http, 'datasets', cand['key'], sha), exts, limit)
        if not files:
            raise Reject('no data file of the target formats within the size cap')
        digests = []
        try:
            for f in files:
                dest = staging / f['path']
                d, nb = await self.http.download(f'{HUB}/datasets/{cand["key"]}/resolve/{sha}/{f["path"]}', dest, limit)
                digests.append(d)
                if dest.name.endswith('.gz'):
                    gunzip(dest, limit * 4)
        except TooLarge as e:
            raise Reject(str(e))
        sha256 = digests[0] if len(digests) == 1 else hashlib.sha256(''.join(digests).encode()).hexdigest()
        return {'version': sha, 'sha256': sha256, 'download_bytes': total,
                'url': f'{HUB}/datasets/{cand["key"]}/tree/{sha}',
                'attribution': f'{cand["key"]} (Hugging Face dataset), revision {sha[:12]}, {cand["license_raw"]}'}


# Model files by format. Config and tokenizer files ride along with any weights format.
CONFIG_FILES = ('config.json', 'generation_config.json', 'preprocessor_config.json')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
                   'vocab.txt', 'vocab.json', 'merges.txt', 'spiece.model', 'sentencepiece.bpe.model',
                   'tokenizer.model', 'source.spm', 'target.spm')
WEIGHTS = {
    'safetensors': lambda p: p.endswith('.safetensors'),
    'pytorch': lambda p: p.endswith(('pytorch_model.bin', '.pt', '.pth')) or re.search(r'pytorch_model-\d+-of-\d+\.bin$', p),
    'tf-checkpoint': lambda p: p.endswith(('tf_model.h5', '.ckpt.index', '.index', 'checkpoint'))
                               or re.search(r'\.data-\d+-of-\d+$', p),
    'onnx': lambda p: p.endswith('.onnx') or p.endswith('.onnx_data'),
}
# ml-model formats -> which weights to take ('' = no weights)
MODEL_FORMATS = {'safetensors': 'safetensors', 'pytorch': 'pytorch', 'tf-checkpoint': 'tf-checkpoint',
                 'onnx': 'onnx', 'hf-download': 'safetensors', 'json-weights': 'safetensors', 'tokenizer': ''}
LIBRARY_FILTER = {'tf-checkpoint': 'tf', 'onnx': 'onnx', 'pytorch': 'pytorch', 'safetensors': 'safetensors'}
MODEL_MAX = 400 << 20
# randomly initialized fixtures for library tests, not trained models
TEST_MODEL = re.compile(r'tiny-random|random-tiny|hf-internal-testing|internal-testing|-testing/|dummy', re.I)
TINY_MAX = 25 << 20            # json-weights: small enough to export to JSON in setup
DTYPE_BYTES = {'F32': 4, 'F16': 2, 'BF16': 2, 'I8': 1, 'U8': 1, 'F64': 8, 'I64': 8, 'I32': 4}


def est_bytes(x):
    st = x.get('safetensors') or {}
    params = st.get('parameters') or {}
    if params:
        return sum(v * DTYPE_BYTES.get(k, 4) for k, v in params.items())
    return None


def model_files(entries, fmts):
    """-> (files to download, total bytes, weights format used) for the target's model formats."""
    paths = {e['path']: e.get('size', 0) for e in entries if e.get('type') == 'file'}
    top = lambda p: '/' not in p  # noqa: E731  (subfolders hold alternative exports; keep the root model)
    want = [MODEL_FORMATS.get(f) for f in fmts if f in MODEL_FORMATS] or ['safetensors']
    weights_fmt = next((w for w in want if w), '')
    chosen = [p for p in paths if top(p) and (p in CONFIG_FILES or p in TOKENIZER_FILES)]
    if weights_fmt:
        order = [weights_fmt] + [w for w in ('safetensors', 'pytorch') if w != weights_fmt]
        for w in order:
            ws = [p for p in paths if (top(p) or w == 'onnx') and WEIGHTS[w](p)]
            if ws:
                chosen += ws
                weights_fmt = w
                break
        else:
            return [], 0, ''
    if 'tokenizer' in fmts and not any(p in TOKENIZER_FILES for p in chosen):
        return [], 0, ''
    return chosen, sum(paths[p] for p in chosen), weights_fmt


class HfModels(Source):
    name = 'hf-model'
    classes = ('ml-model',)

    async def search(self, target, query, n):
        fmts = target.get('formats') or ['safetensors']
        params = [('search', query), ('limit', str(max(30, n * 5))), ('sort', 'downloads'), ('direction', '-1'),
                  ('expand[]', 'safetensors'), ('expand[]', 'cardData'), ('expand[]', 'gated'), ('expand[]', 'sha'),
                  ('expand[]', 'tags'), ('expand[]', 'library_name'), ('expand[]', 'pipeline_tag'),
                  ('expand[]', 'downloads'), ('expand[]', 'siblings')]
        lib = next((LIBRARY_FILTER[f] for f in fmts if f in LIBRARY_FILTER), None)
        if lib:
            params.append(('filter', lib))
        items = await self.http.json(f'{HUB}/api/models', params=params)
        limit = TINY_MAX if 'json-weights' in fmts else MODEL_MAX
        out = []
        for x in items:
            if x.get('gated') or x.get('private') or x.get('disabled') or TEST_MODEL.search(x['id']):
                continue
            est = est_bytes(x)
            if est is not None and 'tokenizer' not in fmts and est > limit:
                continue
            files = [s['rfilename'] for s in x.get('siblings', [])]
            if 'tokenizer' in fmts and not any(Path(f).name in TOKENIZER_FILES for f in files):
                continue
            card = x.get('cardData') or {}
            out.append({'source': self.name, 'key': x['id'], 'url': f'{HUB}/{x["id"]}', 'page_url': f'{HUB}/{x["id"]}',
                        'title': x['id'],
                        'description': f'{x.get("pipeline_tag") or ""} {x.get("library_name") or ""} model; '
                                       f'base {card.get("base_model") or "-"}; datasets {card.get("datasets") or "-"}; '
                                       f'language {card.get("language") or "-"}'[:400],
                        'license_raw': card_license(x), 'size': est,
                        'formats': sorted({k for k, fn in WEIGHTS.items() if any(fn(f) for f in files)}),
                        'meta': {'sha': x.get('sha'), 'downloads': x.get('downloads'),
                                 'pipeline_tag': x.get('pipeline_tag'), 'library': x.get('library_name'),
                                 'n_files': len(files)},
                        'query': query})
        # Exact sizes of the files fetch would take (one tree call per candidate): only safetensors repos carry a
        # parameter count, and the API has no size filter (bert-base TF checkpoints are ~540 MB).
        kept = []
        for c in out:
            if len(kept) >= n * 3:
                break
            try:
                entries = await tree(self.http, 'models', c['key'], c['meta'].get('sha') or 'main')
            except Exception:  # noqa: BLE001  (a repo that vanished or errors: skip it)
                continue
            files, total, wfmt = model_files(entries, fmts)
            if files and total <= limit:
                c['size'] = total
                c['meta']['weights_format'] = wfmt or None
                kept.append(c)
        return kept

    async def fetch(self, cand, staging, target):
        staging = Path(staging)
        fmts = target.get('formats') or ['safetensors']
        limit = TINY_MAX if 'json-weights' in fmts else min(cap(target), MODEL_MAX)
        sha = cand['meta'].get('sha') or 'main'
        entries = await tree(self.http, 'models', cand['key'], sha)
        files, total, wfmt = model_files(entries, fmts)
        if not files:
            raise Reject(f'no files for formats {fmts}')
        if total > limit:
            raise Reject(f'{total} bytes of model files > {limit}')
        digests = {}
        try:
            for p in sorted(files):
                d, _ = await self.http.download(f'{HUB}/{cand["key"]}/resolve/{sha}/{p}', staging / p, limit)
                digests[p] = d
        except TooLarge as e:
            raise Reject(str(e))
        main = next((p for p in sorted(digests) if wfmt and WEIGHTS[wfmt](p)), sorted(digests)[0])
        return {'version': sha, 'sha256': digests[main], 'download_bytes': total, 'weights_format': wfmt or None,
                'url': f'{HUB}/{cand["key"]}/tree/{sha}',
                'attribution': f'{cand["key"]} (Hugging Face model), revision {sha[:12]}, {cand["license_raw"]}'}
