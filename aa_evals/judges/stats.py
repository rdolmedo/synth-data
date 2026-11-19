"""Agreement statistics for the judge study."""
from __future__ import annotations

import random
from collections import Counter


def cohen_kappa(a: list, b: list) -> float | None:
    """Cohen's kappa for two raters over the same items (any hashable labels)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


def kappa_ci(a: list, b: list, n_boot: int = 1000, seed: int = 0) -> list | None:
    n = len(a)
    if n < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            vals.append(k)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * (len(vals) - 1))], 3), round(vals[int(0.975 * (len(vals) - 1))], 3)]


def majority(votes: list) -> object | None:
    """Plurality label; None on a tie for first place."""
    c = Counter(v for v in votes if v is not None).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return None
    return c[0][0]


def rate_ci(k: int, n: int) -> list | None:
    """Wilson 95% interval for k/n, in percent."""
    if n == 0:
        return None
    z, p = 1.96, k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(100 * (mid - half), 1), round(100 * (mid + half), 1)]
