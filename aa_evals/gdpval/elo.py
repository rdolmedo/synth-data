"""Bradley-Terry Elo from pairwise results (AA v2.1: BT/Crowd-BT by MLE, ties as half-wins, an anchored scale).

Players are strings ('expert', or '<run_id>' for a model run). Each observation is (a, b, s, w): a's score s in
{1, 0.5, 0} against b with weight w (a match judged in both orders contributes two half-weight observations).
P(a beats b) = 1 / (1 + 10^((R_b - R_a) / 400)). One player is pinned (`anchor`), the rest are fitted.

Our anchor (#183, approved): the human expert deliverable at 968, which is AA v2's "Human Baseline (1,000)" moved
onto the v2.1 scale (v2.1 re-pinned DeepSeek V4.1 Flash (max) from 1632.1 to 1600; the 18 models in both the
09-15 and 09-20 snapshots moved by a median of -31.4, sd 9). AA's Crowd-BT judge-quality term is not used: one judge.
CIs: a bootstrap over tasks (all of a task's observations together), seed fixed.
"""
from __future__ import annotations

import math

import numpy as np

EXPERT = 'expert'
EXPERT_ELO = 968.0
SCALE = 400.0 / math.log(10.0)


def fit(obs: list[tuple[str, str, float, float]], anchor: str = EXPERT, anchor_elo: float = EXPERT_ELO,
        iters: int = 200, prior: float = 1e-3) -> dict[str, float]:
    """MLE ratings (a tiny ridge `prior` keeps a player who won or lost everything finite)."""
    players = sorted({p for a, b, _, _ in obs for p in (a, b)})
    if anchor not in players:
        players.append(anchor)
    idx = {p: i for i, p in enumerate(players)}
    free = [i for i, p in enumerate(players) if p != anchor]
    if not obs or not free:
        return {anchor: anchor_elo}
    A = np.array([idx[a] for a, _, _, _ in obs])
    B = np.array([idx[b] for _, b, _, _ in obs])
    S = np.array([s for _, _, s, _ in obs], dtype=float)
    W = np.array([w for _, _, _, w in obs], dtype=float)
    theta = np.zeros(len(players))   # natural-log scale, anchor at 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(theta[A] - theta[B])))
        r = W * (S - p)
        g = np.zeros(len(players))
        np.add.at(g, A, r)
        np.add.at(g, B, -r)
        h = W * p * (1 - p)
        H = np.zeros((len(players), len(players)))
        np.add.at(H, (A, A), -h)
        np.add.at(H, (B, B), -h)
        np.add.at(H, (A, B), h)
        np.add.at(H, (B, A), h)
        g -= prior * theta
        H -= prior * np.eye(len(players))
        gf, Hf = g[free], H[np.ix_(free, free)]
        step = np.linalg.solve(Hf, -gf)
        theta[free] += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {p: anchor_elo + SCALE * theta[idx[p]] for p in players}


def bootstrap(obs_by_task: dict[str, list], n: int = 1000, seed: int = 731, **kw) -> dict[str, tuple[float, float]]:
    """95% percentile CI per player from resampling tasks with replacement."""
    rng = np.random.default_rng(seed)
    tasks = sorted(obs_by_task)
    draws: dict[str, list[float]] = {}
    for _ in range(n):
        pick = rng.integers(0, len(tasks), len(tasks))
        obs = [o for i in pick for o in obs_by_task[tasks[i]]]
        for p, r in fit(obs, **kw).items():
            draws.setdefault(p, []).append(r)
    return {p: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for p, v in draws.items()}


def elo_from_winrate(w: float, opponent_elo: float = EXPERT_ELO) -> float:
    w = min(max(w, 1e-6), 1 - 1e-6)
    return opponent_elo + 400 * math.log10(w / (1 - w))


def winrate_from_elo(r: float, opponent_elo: float = EXPERT_ELO) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - r) / 400))
