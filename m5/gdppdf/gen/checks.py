"""Problem-side checks (format, grounding, leakage, repeats within a document). No difficulty or correctness screens.

Each check returns a reason string when the request is rejected; flags are kept as metadata.
"""
import re

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'.,%$/-]*")
ANSWERY = re.compile(r'\b(the answer is|correct answer|answer key|expected answer|rubric|grading|the solution is)\b', re.I)
META = re.compile(r'\b(slot \d|pitfall|trap|task family|this request is designed|for the dataset|the assistant should '
                  r'notice)\b', re.I)
VISUAL = re.compile(r'\b(in|on|from) the (chart|graph|figure|photo|map|diagram|drawing|image)s?\b', re.I)


def words(t):
    return WORD.findall(t.lower())


def ngrams(ws, n=8):
    return {' '.join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def check_request(req, n_pages, doc_ngrams8):
    """-> (reject_reason or None, flags dict)."""
    p = (req.get('prompt') or '').strip()
    flags = {}
    if not p:
        return 'empty', flags
    ws = words(p)
    flags['words'] = len(ws)
    if len(ws) < 8:
        return 'too_short', flags
    if len(ws) > 1600:
        return 'too_long', flags
    if ANSWERY.search(p):
        return 'answer_or_rubric_language', flags
    if META.search(p):
        return 'meta_language', flags
    ascii_share = sum(c.isascii() for c in p) / len(p)
    if ascii_share < 0.9:
        return 'non_english', flags
    pages = [x for x in (req.get('pages_used') or []) if isinstance(x, int)]
    flags['pages_used'] = sorted(set(pages))
    if any(x < 1 or x > n_pages for x in pages):
        flags['bad_pages'] = True
        pages = [x for x in pages if 1 <= x <= n_pages]
    if n_pages >= 6 and len(set(pages)) < 2:
        return 'single_location', flags
    flags['page_span'] = (max(pages) - min(pages)) if pages else 0
    g = ngrams(ws)
    if g:
        copied = len(g & doc_ngrams8) / len(g)
        flags['doc_8gram_share'] = round(copied, 3)
        if copied > 0.5:
            return 'copies_document', flags
    if VISUAL.search(p):
        flags['mentions_visual'] = True
    return None, flags


def tfidf_dupes(prompts, threshold=0.6):
    """Indices to drop: a later request whose TF-IDF cosine with an earlier kept one exceeds the threshold."""
    if len(prompts) < 2:
        return set()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    m = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True).fit_transform(prompts)
    sim = cosine_similarity(m)
    drop = set()
    for j in range(len(prompts)):
        for i in range(j):
            if i not in drop and sim[i, j] > threshold:
                drop.add(j)
                break
    return drop
