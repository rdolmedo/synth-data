"""The judge each judged benchmark grades with, pinned by arm name (aa_evals/judges/arms.py; audit 1 on #41).

A runner given no --judge-arm (and no explicit --judge) uses its benchmark's pin here, and records the arm as
judge.arm in config.json and summary.json. The pin changes only by a decision recorded on the benchmark's issue:
- until study s3 (#41) reports: qwen38-27b-medium-pin everywhere (the M3.2 default, #175);
- then: s3's per-benchmark recommendation;
- final: the pick against M3.5's DeepSeek gold judge (#178).
"""
from __future__ import annotations

from .arms import ARMS

BENCH_JUDGE_ARM = {
    'omniscience': 'qwen38-27b-medium-pin',   # #31
    'hle': 'qwen38-27b-medium-pin',           # #33
    'gdppdf': 'qwen38-27b-medium-pin',        # #34
}

assert all(a in ARMS for a in BENCH_JUDGE_ARM.values())


def default_arm(bench: str) -> str:
    return BENCH_JUDGE_ARM[bench]
