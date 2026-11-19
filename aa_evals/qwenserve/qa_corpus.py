"""QA-shaped replay corpus for the Qwen3.8-27B serving study (#70): single-turn HLE-Diamond questions as the AA
eval sends them (aa_evals.hle.prompts.build_messages, AA's by-type system prompt, images inline), half at
reasoning_effort medium and half at xhigh (the overnight rounds run both), sampled as serving:aa-qwen's registry.

  python -I aa_evals/qwenserve/qa_corpus.py --parquet <hle-diamond test.parquet> --out /data/aai/aa_evals/runs/qwenserve/qa-corpus

Writes one request body per question under <out>/bodies/ and <out>/corpus.json, a synth.bench corpus whose
trajectories are one request each (synth.tune `corpus_file`). Eval data: stays on /data, never in git or training.
max_tokens is capped (default 32,768) so a replay window reaches a steady state in minutes (deviation, see README).
"""
import argparse
import hashlib
import json
from pathlib import Path

SYSTEM_EXACT_ANSWER = ("Your response should be in the following format:\n"
                       "Explanation: {your explanation for your final answer}\n"
                       "Exact Answer: {your succinct, final answer}\n"
                       "Confidence: {your confidence score between 0% and 100% for your answer}")
SYSTEM_MC = ("Your response should be in the following format:\n"
             "Explanation: {your explanation for your answer choice}\n"
             "Answer: {your chosen answer}\n"
             "Confidence: {your confidence score between 0% and 100% for your answer}")
SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'presence_penalty': 0.0,
            'repetition_penalty': 1.0}


def body(item, effort, max_tokens):
    system = SYSTEM_EXACT_ANSWER if item['answer_type'] == 'exactMatch' else SYSTEM_MC
    content = [{'type': 'text', 'text': item['question']}]
    if item.get('image'):
        content.append({'type': 'image_url', 'image_url': {'url': item['image']}})
    return {'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            **SAMPLING, 'max_tokens': max_tokens, 'reasoning_effort': effort,
            'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True, 'reasoning_effort': effort},
            'seed': int(hashlib.sha256(item['id'].encode()).hexdigest()[:8], 16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-tokens', type=int, default=32768)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    rows = pq.read_table(a.parquet, columns=['id', 'question', 'image', 'answer_type']).to_pylist()
    out = Path(a.out)
    (out / 'bodies').mkdir(parents=True, exist_ok=True)
    trajectories = []
    for item in sorted(rows, key=lambda r: r['id']):
        effort = 'xhigh' if int(hashlib.sha256(item['id'].encode()).hexdigest(), 16) % 2 else 'medium'
        path = out / 'bodies' / f'{item["id"]}.json'
        path.write_text(json.dumps(body(item, effort, a.max_tokens)))
        trajectories.append({'run': item['id'], 'requests': [str(path)], 'gaps': [None], 'effort': effort,
                             'image': bool(item.get('image'))})
    (out / 'corpus.json').write_text(json.dumps({'source': a.parquet, 'max_tokens': a.max_tokens,
                                                 'trajectories': trajectories}))
    print(f'{len(trajectories)} questions ({sum(t["image"] for t in trajectories)} with images, '
          f'{sum(t["effort"] == "xhigh" for t in trajectories)} xhigh) -> {out}')


if __name__ == '__main__':
    main()
