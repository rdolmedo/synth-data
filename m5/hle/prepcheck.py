"""Prep dry run: sft/prepare_data.py (origin/sft) on the exported shards, CPU only, output in job scratch.

    python -m m5.hle.prepcheck --prep CODE/prepare_data.py --inputs 'E/traces-*.jsonl.gz' --out E/prepcheck \
        [--workers 8] [--qwen-code SFT_CODE]

Only the reports (stats.json, report.md, sample_rendered.txt) are copied to --out; the packed arrays are discarded.
With --qwen-code, also m5.hle.qwencheck (the SFT branch's Qwen3.5 renderer) under --qwen-python -> --out/qwen.json.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', required=True)
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='/data/aai/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--qwen-code', help='dir holding sft_qwen35/ (data.py, chat_template.jinja) from origin/sft')
    ap.add_argument('--qwen-python', default='/opt/aai/venvs/qwen35sft/bin/python')
    ap.add_argument('--qwen-model', default='/data/aai/models/qwen3.5-4b')
    ap.add_argument('--skip-prep', action='store_true')
    a = ap.parse_args()
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / 'm5hle-prep'
    shutil.rmtree(scratch, ignore_errors=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    if a.qwen_code:
        q = subprocess.run([a.qwen_python, '-m', 'm5.hle.qwencheck', '--code', a.qwen_code, '--inputs', a.inputs,
                            '--out', str(out / 'qwen.json'), '--model', a.qwen_model, '--workers', str(a.workers)],
                           env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
        rc = rc or q.returncode
    if a.skip_prep:
        raise SystemExit(rc)
    r = subprocess.run([sys.executable, a.prep, '--inputs', a.inputs, '--out', str(scratch), '--model', a.model,
                        '--workers', str(a.workers), '--calib-samples', '0'])
    for name in ('stats.json', 'report.md', 'sample_rendered.txt'):
        if (scratch / name).exists():
            shutil.copy(scratch / name, out / name)
            print('copied', out / name, flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    raise SystemExit(rc or r.returncode)


if __name__ == '__main__':
    main()
