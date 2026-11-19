from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from nemotron_admission_report import report


def rows():
    return [{"low": low, "seed": seed, "measure_seconds": 900, "generated_tokens": 900 * tps,
             "preemptions": 0, "errors": 0, "warmup_errors": 0, "warmup_preemptions": 0,
             "resume_transitions": 1, "prompt_tokens": 1000, "cached_prompt_tokens": 800,
             "ttft_sum": 3, "ttft_count": 10}
            for seed in (0, 1) for low, tps in [(0.60, 4000), (0.75, 3500), (0.85, 3000)]]


def test_incomplete_or_unstable_experiment_cannot_recommend_threshold():
    assert report(rows()[:3])["recommended_low_watermark"] is None
    data = rows()
    for row in data:
        row["preemptions"] = 1
    assert report(data)["recommended_low_watermark"] is None


def test_paired_improvement_requires_consistency_and_safe_startup():
    data = rows()
    assert report(data)["recommended_low_watermark"] == 0.60
    data[0]["warmup_preemptions"] = 1
    assert report(data)["recommended_low_watermark"] is None
    data[0]["warmup_preemptions"] = 0
    data[0]["generated_tokens"] = 900 * 3000
    assert report(data)["recommended_low_watermark"] is None
