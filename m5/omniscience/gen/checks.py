"""Problem-side checks for generated questions (format and grounding only: no difficulty screen, no answer check).

parse(): the generator's JSON. check(): reasons a question is rejected (empty list = accepted).
Dedup: an online near-duplicate index (word 3-gram Jaccard among questions sharing a rare token).
"""
import collections
import math
import json
import re
import unicodedata

REF_PASSAGE = re.compile(r'\b(the|this|that|above|given|provided|following) (passage|text|excerpt|article|document|'
                         r'snippet|source|paragraph|abstract|extract)\b|\baccording to the (author|study|passage|text|'
                         r'article|document|report)\b|\bmentioned above\b|\bdescribed above\b', re.I)
MAX_Q, MIN_Q, MAX_A = 700, 30, 160
STOP = set('a an the of in on for to and or by with at from as is are was were be been which what who whom whose when '
           'where how why that this these those it its into under over per than then there their his her they them '
           'did does do has have had not no nor but if so such each any all one two first'.split())


def nfkc(s):
    return unicodedata.normalize('NFKC', s or '')


def norm(s):
    return re.sub(r'[^0-9a-z]+', ' ', nfkc(s).lower()).strip()


def parse(content: str):
    """The last JSON object in the reply, or None."""
    s = content.strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    try:
        return json.loads(s)
    except ValueError:
        pass
    for m in reversed(list(re.finditer(r'\{', s))):
        try:
            d, _ = json.JSONDecoder().raw_decode(s[m.start():])
            if isinstance(d, dict) and ('status' in d or 'flags' in d):
                return d
        except ValueError:
            continue
    # lenient fallback: unescaped quotes inside values (keys appear in the order the prompt asks for)
    keys = ['status', 'question', 'answer', 'evidence', 'answer_type']
    m = re.search(r'"status"\s*:\s*"ok".*?"question"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*,\s*'
                  r'"evidence"\s*:\s*"(.*?)"\s*,\s*"answer_type"\s*:\s*"(.*?)"\s*\}', s, re.S)
    if m:
        return {'status': 'ok', 'question': m.group(1), 'answer': m.group(2), 'evidence': m.group(3),
                'answer_type': m.group(4), '_lenient': True}
    if re.search(r'"status"\s*:\s*"skip"', s):
        return {'status': 'skip', 'reason': 'unparsed skip'}
    return None


def _in(needle, hay):
    n, h = norm(needle), norm(hay)
    return bool(n) and f' {n} ' in f' {h} '


def check(d: dict, passage_text: str) -> list[str]:
    bad = []
    q, a, ev = (d.get('question') or '').strip(), (d.get('answer') or '').strip(), (d.get('evidence') or '').strip()
    if not q or not a:
        return ['missing_fields']
    if not (MIN_Q <= len(q) <= MAX_Q):
        bad.append('question_length')
    if len(a) > MAX_A:
        bad.append('answer_length')
    if REF_PASSAGE.search(q):
        bad.append('refers_to_passage')
    if re.search(r'\b\w+\.(htm|html|txt|xml|md|rst|pdf)\b|\b\d{7,}_\d{4}\b', q):
        bad.append('file_reference')
    if q.count('?') > 2:
        bad.append('multiple_questions')
    na = norm(a)
    if len(na) >= 3 and _in(a, q):
        bad.append('answer_in_question')
    # grounding: evidence copied from the passage (allowing whitespace/punctuation drift) and the answer in it
    if not ev:
        bad.append('no_evidence')
    else:
        ne, npass = norm(ev), norm(passage_text)
        if ne not in npass:
            # tolerate small edits: 80% of the evidence's word 5-grams must occur in the passage
            w = ne.split()
            grams = [' '.join(w[i:i + 5]) for i in range(max(1, len(w) - 4))]
            hit = sum(g in npass for g in grams) / max(1, len(grams))
            if hit < 0.8:
                bad.append('evidence_not_in_passage')
    if na and not (_in(a, ev) or _in(a, passage_text) or _answer_tokens_in(a, passage_text)):
        bad.append('answer_not_in_passage')
    return bad


def _answer_tokens_in(a, text):
    """Lenient grounding for formulae/numbers/reformatted answers: most answer tokens appear in the passage."""
    toks = [t for t in norm(a).split() if t not in STOP]
    if not toks:
        return False
    h = f' {norm(text)} '
    return sum(f' {t} ' in h for t in toks) / len(toks) >= 0.75


def tokens(s):
    return [t for t in norm(s).split() if t not in STOP]


class DedupIndex:
    """Online near-duplicate detection over accepted questions."""

    def __init__(self, thresh=0.5, leaf_cos=0.78):
        self.leaf_cos = leaf_cos
        self.by_leaf = collections.defaultdict(list)   # leaf -> [(pid, Counter of content words)]
        self.df = collections.Counter()
        self.post = collections.defaultdict(list)
        self.grams = []
        self.ids = []
        self.thresh = thresh
        self.exact = {}

    @staticmethod
    def _grams(toks):
        return {' '.join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))}

    @staticmethod
    def _cos(a, b):
        n = sum(v * b[k] for k, v in a.items() if k in b)
        return n / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()) or 1)

    def find(self, q, leaf=None):
        """-> id of a near-duplicate already indexed, or None. Within the same leaf, a content-word cosine >= leaf_cos
        also counts (the same fact restated from another document; calibrated on the dense-leaf prototype)."""
        n = norm(q)
        if n in self.exact:
            return self.exact[n]
        if leaf is not None:
            bag = collections.Counter(tokens(q))
            for pid, b in self.by_leaf[leaf]:
                if self._cos(bag, b) >= self.leaf_cos:
                    return pid
        toks = tokens(q)
        g = self._grams(toks)
        rare = sorted(set(toks), key=lambda t: self.df[t])[:4]
        cands = set()
        for t in rare:
            cands.update(self.post[t][-5000:])
        for c in cands:
            h = self.grams[c]
            j = len(g & h) / max(1, len(g | h))
            if j >= self.thresh:
                return self.ids[c]
        return None

    def add(self, pid, q, leaf=None):
        toks = tokens(q)
        if leaf is not None:
            self.by_leaf[leaf].append((pid, collections.Counter(toks)))
        k = len(self.ids)
        self.ids.append(pid)
        self.grams.append(self._grams(toks))
        self.exact[norm(q)] = pid
        for t in set(toks):
            self.df[t] += 1
            self.post[t].append(k)
