"""Build the single user message AA sends for a GDP.pdf task, adapted to one endpoint's limits.

AA (methodology page, GDP.pdf):
- Every model receives the full LiteParse-extracted text of every page. Models with image input also get
  page images, sent first in page order, then the task and the text, in one user message.
- Page images are rendered at 150 DPI and "may be reduced to a 72 DPI minimum when context or payload limits
  require"; opaque pages may be converted from PNG to JPEG.
- When the endpoint caps images per request, pages are combined into labelled composites, two pages per
  image, or up to four where the cap is tighter. Beyond four pages per image, images cover only the leading
  pages and the rest stay in the text. The prompt says when composites are used and where coverage stops.
- If images still don't fit, the complete extracted text is sent alone. The text is never truncated.

The prompt wording itself is not published; ours is below (`render_text`).

Token budgeting: prompt tokens = text tokens (endpoint /tokenize) + image tokens (the model's own image rule)
+ a margin, and we require prompt + max_tokens <= the endpoint's max_model_len. This matters on vLLM's
Nemotron VL processor, which silently shrinks every image when the images overflow the context instead of
rejecting the request.
"""
import base64
import io
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

DPI_LADDER = (150, 120, 96, 72)
COMPOSITE_STEPS = (2, 4)
LABEL_PT = 14            # composite label height in points (scaled by DPI)
GUTTER_PT = 6
JPEG_QUALITY = 90
TOKEN_MARGIN = 512       # chat template, the coverage note
PER_IMAGE_OVERHEAD = 8   # with >1 image, Nemotron's template labels each one '<image N>' (+5-6 tokens; measured)
VLLM_DECODE_LIMIT = 715827880   # vLLM's default VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES; it counts W*H*4 per image


# ---------------------------------------------------------------------------------------------------------
# Image token rules: how many prompt tokens one image of (w, h) pixels costs on a given model.

def nemotron_vl_tokens(w, h, patch=16, min_patches=1024, max_patches=13312, shuffle=2):
    """vLLM nano_nemotron_vl DynamicResolutionImageTiler.process_media with an unconstrained context budget:
    ceil-ish patch grid, scaled down to max_patches, up to min_patches, rounded to even, 2x2 pixel shuffle."""
    ph = round(h / patch + 0.5)
    pw = round(w / patch + 0.5)
    budget = max_patches
    f = min(math.sqrt(budget / (ph * pw)), 1.0)
    th, tw = math.floor(f * ph), math.floor(f * pw)
    if budget > min_patches and th * tw < min_patches:
        up = math.sqrt(min_patches / (th * tw))
        th, tw = math.ceil(up * th), math.ceil(up * tw)
    if th % shuffle:
        th = th + (shuffle - th % shuffle) if (th + shuffle - th % shuffle) * tw <= budget else max(shuffle, th - th % shuffle)
    if tw % shuffle:
        tw = tw + (shuffle - tw % shuffle) if th * (tw + shuffle - tw % shuffle) <= budget else max(shuffle, tw - tw % shuffle)
    return tw * th // (shuffle * shuffle) + 2   # + <img>, </img> (2,082 per 150 DPI letter page, as /tokenize counts)


def qwen_vl_tokens(w, h, patch=28, max_pixels=16384 * 28 * 28):
    """Qwen2/3-VL style: one token per 28x28 block after smart-resize to max_pixels (a conservative default)."""
    px = w * h
    if px > max_pixels:
        s = math.sqrt(max_pixels / px)
        w, h = w * s, h * s
    return math.ceil(w / patch) * math.ceil(h / patch) + 2


class ProbeRule:
    """Any vLLM endpoint: the exact token cost of one image of (w, h), measured once per size with /tokenize on a
    one-image chat message (minus the same message without the image). Dynamic-resolution processors price an
    image by its size alone, as long as it doesn't overflow the context on its own."""

    def __init__(self, root_url, served_name, headers=None):
        import threading
        self.root, self.model, self.headers = root_url, served_name, headers or {}
        self.cache, self.lock, self.base = {}, threading.Lock(), None

    def _count(self, content):
        import httpx
        for attempt in range(5):
            try:
                r = httpx.post(f'{self.root}/tokenize', headers=self.headers, timeout=120, trust_env=True,
                               json={'model': self.model, 'messages': [{'role': 'user', 'content': content}],
                                     'add_generation_prompt': True})
                if r.status_code == 200:
                    return int(r.json()['count'])
                err = f'HTTP {r.status_code}: {r.text[:200]}'
            except (httpx.HTTPError, ValueError, KeyError) as e:
                err = str(e)
            import time
            time.sleep(2 + 3 * attempt)
        raise RuntimeError(f'/tokenize image probe failed: {err}')

    def __call__(self, w, h):
        key = (w, h)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        from PIL import Image
        if self.base is None:
            self.base = self._count([{'type': 'text', 'text': 'x'}])
        buf = io.BytesIO()
        Image.new('RGB', (w, h), (255, 255, 255)).save(buf, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        n = self._count([{'type': 'image_url', 'image_url': {'url': url}}, {'type': 'text', 'text': 'x'}]) - self.base
        with self.lock:
            self.cache[key] = n
        return n


IMAGE_RULES = {'nemotron_vl': nemotron_vl_tokens, 'qwen_vl': qwen_vl_tokens, 'probe': None}


def image_rule_for(served_name, kind='serving'):
    n = (served_name or '').lower()
    if 'nemotron' in n:
        return 'nemotron_vl'
    return 'probe' if kind != 'openai' else 'qwen_vl'


# ---------------------------------------------------------------------------------------------------------
# Rendering (LiteParse's own renderer, one process pool; PNG bytes are flattened to RGB on white).

_LP = {}


def _render_pages(pdf, dpi, pages):
    """Worker: [(page, opaque, png_rgb_bytes)] for 1-indexed `pages` of `pdf` at `dpi`."""
    from liteparse import LiteParse
    from PIL import Image
    lp = _LP.get(dpi)
    if lp is None:
        lp = _LP[dpi] = LiteParse(dpi=float(dpi), ocr_enabled=False, quiet=True)
    out = []
    for shot in lp.screenshot(pdf, page_numbers=list(pages)):
        im = Image.open(io.BytesIO(shot.image_bytes))
        opaque = True
        if im.mode in ('RGBA', 'LA', 'P'):
            im = im.convert('RGBA')
            opaque = im.getchannel('A').getextrema()[0] == 255
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel('A'))
            im = bg
        elif im.mode != 'RGB':
            im = im.convert('RGB')
        buf = io.BytesIO()
        im.save(buf, format='PNG', optimize=False)
        out.append((shot.page_num, opaque, buf.getvalue()))
    return out


class Renderer:
    """Renders page images in a process pool, caching one (pdf, dpi) at a time per task."""

    def __init__(self, workers=4):
        import multiprocessing
        # spawn, not fork: the parent runs an asyncio loop with worker threads
        self.pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
                     if workers > 1 else None)
        self.workers = max(1, workers)

    def render(self, pdf, dpi, n_pages):
        pages = list(range(1, n_pages + 1))
        if not self.pool:
            got = _render_pages(str(pdf), dpi, pages)
        else:
            chunks = [pages[i::self.workers] for i in range(self.workers) if pages[i::self.workers]]
            got = [x for part in self.pool.map(_render_pages, [str(pdf)] * len(chunks), [dpi] * len(chunks), chunks)
                   for x in part]
        got.sort(key=lambda x: x[0])
        if [g[0] for g in got] != pages:
            raise RuntimeError(f'rendered pages {[g[0] for g in got][:5]}... != 1..{n_pages} for {pdf}')
        return [Page(num=p, opaque=o, png=b) for p, o, b in got]

    def close(self):
        if self.pool:
            self.pool.shutdown(cancel_futures=True)


@dataclass
class Page:
    num: int
    opaque: bool
    png: bytes


@dataclass
class Img:
    pages: list          # 1-indexed page numbers shown in this image
    width: int
    height: int
    opaque: bool
    data: bytes
    mime: str = 'image/png'


def _size(png):
    from PIL import Image
    return Image.open(io.BytesIO(png)).size


def compose(pages, k, dpi):
    """Group consecutive pages k at a time into labelled composites (k=2: side by side; k=4: 2x2 grid)."""
    from PIL import Image, ImageDraw, ImageFont
    if k == 1:
        out = []
        for p in pages:
            w, h = _size(p.png)
            out.append(Img(pages=[p.num], width=w, height=h, opaque=p.opaque, data=p.png))
        return out
    cols, rows = (2, 1) if k == 2 else (2, 2)
    label_h = max(12, round(LABEL_PT * dpi / 72))
    gutter = max(2, round(GUTTER_PT * dpi / 72))
    font = ImageFont.load_default(size=max(10, round(label_h * 0.75)))
    out = []
    for i in range(0, len(pages), k):
        group = pages[i:i + k]
        ims = [Image.open(io.BytesIO(p.png)).convert('RGB') for p in group]
        cw = max(im.width for im in ims)
        ch = max(im.height for im in ims) + label_h
        canvas = Image.new('RGB', (cols * cw + (cols - 1) * gutter, rows * ch + (rows - 1) * gutter), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for j, (p, im) in enumerate(zip(group, ims)):
            x, y = (j % cols) * (cw + gutter), (j // cols) * (ch + gutter)
            draw.rectangle([x, y, x + cw - 1, y + label_h - 1], fill=(225, 225, 225))
            draw.text((x + label_h // 3, y + label_h // 8), f'Page {p.num}', fill=(0, 0, 0), font=font)
            canvas.paste(im, (x, y + label_h))
        for c in range(1, cols):
            draw.rectangle([c * cw + (c - 1) * gutter, 0, c * (cw + gutter) - 1, canvas.height], fill=(128, 128, 128))
        for r in range(1, rows):
            draw.rectangle([0, r * ch + (r - 1) * gutter, canvas.width, r * (ch + gutter) - 1], fill=(128, 128, 128))
        buf = io.BytesIO()
        canvas.save(buf, format='PNG')
        out.append(Img(pages=[p.num for p in group], width=canvas.width, height=canvas.height,
                       opaque=all(p.opaque for p in group), data=buf.getvalue()))
    return out


def to_jpeg(img):
    from PIL import Image
    if not img.opaque or img.mime == 'image/jpeg':
        return img
    buf = io.BytesIO()
    Image.open(io.BytesIO(img.data)).convert('RGB').save(buf, format='JPEG', quality=JPEG_QUALITY)
    return Img(pages=img.pages, width=img.width, height=img.height, opaque=True, data=buf.getvalue(), mime='image/jpeg')


def payload_bytes(imgs):
    return sum(4 * math.ceil(len(i.data) / 3) + 32 for i in imgs)


# ---------------------------------------------------------------------------------------------------------
# Prompt text.

def coverage_note(plan, n_pages, pdf_name):
    if plan.mode == 'text':
        return None
    if plan.k == 1:
        s = (f'The {plan.n_images} images above are the pages of the attached document "{pdf_name}", in order '
             f'(image i is page i).')
    else:
        s = (f'The {plan.n_images} images above show the pages of the attached document "{pdf_name}" in order, '
             f'combined {plan.k} consecutive pages per image; each page is labelled with its page number.')
    if plan.covered < n_pages:
        s += (f' The images cover pages 1-{plan.covered} only; pages {plan.covered + 1}-{n_pages} are available '
              f'only as extracted text below.')
    return s


def render_text(task_prompt, pages_text, pdf_name, note=None):
    """The text part of the user message: coverage note, the task, then the full per-page extracted text."""
    n = len(pages_text)
    parts = []
    if note:
        parts.append(note + '\n\n')
    parts.append(task_prompt.rstrip() + '\n\n')
    parts.append(f'Extracted text of the attached document "{pdf_name}" ({n} page{"s" if n != 1 else ""}), '
                 f'produced by a PDF parser with OCR; it may contain recognition errors and layout artifacts.\n')
    for i, t in enumerate(pages_text, 1):
        parts.append(f'\n--- Page {i} ---\n{t.rstrip()}\n')
    return ''.join(parts)


# ---------------------------------------------------------------------------------------------------------
# The adaptation ladder.

@dataclass
class Limits:
    max_model_len: int          # endpoint context (prompt + output)
    max_tokens: int             # output budget requested
    max_images: int             # images per request (0 = text-only model)
    max_payload_mb: float = 100.0
    max_decoded_bytes: int = VLLM_DECODE_LIMIT   # server cap on decoded media per request (0 = none)
    image_rule: str = 'nemotron_vl'
    image_tokens: object = None  # a callable (w, h) -> tokens; overrides image_rule (ProbeRule)


@dataclass
class Plan:
    mode: str                   # 'images' or 'text'
    k: int = 1                  # pages per image
    dpi: int = 0
    covered: int = 0            # pages 1..covered appear in images
    n_images: int = 0
    jpeg: bool = False
    image_tokens: int = 0
    text_tokens: int = 0
    est_prompt_tokens: int = 0
    payload_mb: float = 0.0
    reason: str = ''
    tried: list = field(default_factory=list)

    def describe(self):
        return {k: v for k, v in self.__dict__.items()}


def composite_factor(n_pages, max_images):
    """(k, covered): pages per image and how many leading pages the images can cover."""
    if n_pages <= max_images:
        return 1, n_pages
    for k in COMPOSITE_STEPS:
        if math.ceil(n_pages / k) <= max_images:
            return k, n_pages
    k = COMPOSITE_STEPS[-1]
    return k, max_images * k


def plan_and_build(task, pages_text, limits, base_text_tokens, renderer, min_dpi=None, start_dpi=None):
    """Walk AA's ladder and return (plan, imgs, text).

    `base_text_tokens`: the endpoint's token count of render_text(..., note=None); the coverage note adds a
    few dozen tokens, estimated at 3 chars per token. `start_dpi` resumes below a DPI the endpoint rejected;
    `min_dpi` is the floor (72)."""
    n = len(pages_text)
    rule = limits.image_tokens or IMAGE_RULES[limits.image_rule]
    tried = []
    text_cache = {}

    def text_for(note):
        if note not in text_cache:
            t = render_text(task.prompt, pages_text, task.pdf_name, note)
            text_cache[note] = (t, base_text_tokens + (len(note) // 3 + 2 if note else 0))
        return text_cache[note]

    if limits.max_images > 0:
        k, covered = composite_factor(n, limits.max_images)
        ladder = [d for d in DPI_LADDER if (start_dpi is None or d <= start_dpi) and d >= (min_dpi or DPI_LADDER[-1])]
        for dpi in ladder:
            pages = renderer.render(task.pdf_path, dpi, n)[:covered]
            imgs = compose(pages, k, dpi)
            img_tok = sum(rule(i.width, i.height) for i in imgs)
            plan = Plan(mode='images', k=k, dpi=dpi, covered=covered, n_images=len(imgs))
            text, text_tok = text_for(coverage_note(plan, n, task.pdf_name))
            est = text_tok + img_tok + TOKEN_MARGIN + (PER_IMAGE_OVERHEAD * len(imgs) if len(imgs) > 1 else 0)
            decoded = sum(i.width * i.height * 4 for i in imgs)
            fits = (est + limits.max_tokens <= limits.max_model_len
                    and (not limits.max_decoded_bytes or decoded <= limits.max_decoded_bytes))
            pay = payload_bytes(imgs) / 1e6
            jpeg = False
            if fits and pay > limits.max_payload_mb:
                imgs = [to_jpeg(i) for i in imgs]
                jpeg, pay = True, payload_bytes(imgs) / 1e6
            tried.append({'dpi': dpi, 'k': k, 'est_prompt_tokens': est, 'payload_mb': round(pay, 1),
                          'decoded_mb': round(decoded / 2 ** 20), 'fits': fits, 'jpeg': jpeg})
            if fits and pay <= limits.max_payload_mb:
                plan.jpeg, plan.image_tokens, plan.text_tokens = jpeg, img_tok, text_tok
                plan.est_prompt_tokens, plan.payload_mb, plan.tried = est, round(pay, 1), tried
                plan.reason = 'fits' if dpi == DPI_LADDER[0] and k == 1 and not jpeg else 'adapted'
                return plan, imgs, text
    text, text_tok = text_for(None)
    est = text_tok + TOKEN_MARGIN
    plan = Plan(mode='text', text_tokens=text_tok, est_prompt_tokens=est, tried=tried,
                reason='text-only model' if limits.max_images <= 0 else 'images do not fit')
    if est + limits.max_tokens > limits.max_model_len:
        plan.reason = 'input_too_long'
    return plan, [], text


def message_content(imgs, text):
    """OpenAI chat content: images first in page order, then the text."""
    parts = [{'type': 'image_url', 'image_url': {'url': f'data:{i.mime};base64,{base64.b64encode(i.data).decode()}'}}
             for i in imgs]
    parts.append({'type': 'text', 'text': text})
    return parts
