"""Official HLE prompts (centerforaisafety/hle) and the request messages built from them.

Pinned source: centerforaisafety/hle hle_eval/*.py. The two answer-format prompts are from commit
136e12e835 (before 67b3251 "Unify system prompt", 2025-06-06); AA's methodology quotes this
two-variant form. The judge prompt and its response schema are unchanged at HEAD 22ed307.
"""

SYSTEM_EXACT_ANSWER = ("Your response should be in the following format:\n"
                       "Explanation: {your explanation for your final answer}\n"
                       "Exact Answer: {your succinct, final answer}\n"
                       "Confidence: {your confidence score between 0% and 100% for your answer}")

SYSTEM_MC = ("Your response should be in the following format:\n"
             "Explanation: {your explanation for your answer choice}\n"
             "Answer: {your chosen answer}\n"
             "Confidence: {your confidence score between 0% and 100% for your answer}")

# HEAD's single prompt (since 2025-06-06) is the MC wording verbatim.
SYSTEM_UNIFIED = SYSTEM_MC

# System-prompt policies:
#   by_type   AA's description: exact-answer questions get SYSTEM_EXACT_ANSWER, multiple choice SYSTEM_MC.
#   code      what the official two-prompt code actually sent: it tests answer_type == 'exact_match', but the
#             dataset's value is 'exactMatch', so every question got SYSTEM_MC (same text as `unified`).
#   unified   the repo's HEAD: one prompt for all.
SYSTEM_POLICIES = ('by_type', 'code', 'unified')


def system_prompt(item, policy='by_type'):
    if policy == 'by_type':
        return SYSTEM_EXACT_ANSWER if item['answer_type'] == 'exactMatch' else SYSTEM_MC
    if policy == 'code':
        return SYSTEM_EXACT_ANSWER if item['answer_type'] == 'exact_match' else SYSTEM_MC
    if policy == 'unified':
        return SYSTEM_UNIFIED
    raise ValueError(policy)


def build_messages(item, *, policy='by_type', text_only=False, system_role='system'):
    """The chat messages for one question, as run_model_predictions.py builds them.

    The image (a data URI in the dataset) follows the question text. `text_only` drops it, as AA does
    for models without image input. `system_role='user'` prepends the prompt to the user turn, AA's
    handling of models without system-prompt support.
    """
    prompt = system_prompt(item, policy)
    text = item['question']
    if system_role == 'user':
        text = prompt + '\n\n' + text
    content = [{'type': 'text', 'text': text}]
    if item['image'] and not text_only:
        content.append({'type': 'image_url', 'image_url': {'url': item['image']}})
    if system_role == 'user':
        return [{'role': 'user', 'content': content}]
    if system_role != 'system':
        raise ValueError(system_role)
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': content}]


JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""

# run_judge_results.py's ExtractedAnswer schema, with confidence bounded to the prompt's 0..100 contract.
JUDGE_SCHEMA = {
    'type': 'object',
    'properties': {
        'extracted_final_answer': {'type': 'string'},
        'reasoning': {'type': 'string'},
        'correct': {'type': 'string', 'enum': ['yes', 'no']},
        # Finite values enforce the prompt's 0..100 contract during guided decoding.
        # Unbounded integers can decode thousands of trailing zeros instead of closing.
        'confidence': {'type': 'integer', 'enum': list(range(101))},
        'strict': {'type': 'boolean', 'enum': [True]},
    },
    'required': ['extracted_final_answer', 'reasoning', 'correct', 'confidence', 'strict'],
    'additionalProperties': False,
}
JUDGE_RESPONSE_FORMAT = {'type': 'json_schema',
                         'json_schema': {'name': 'ExtractedAnswer', 'schema': JUDGE_SCHEMA, 'strict': True}}


def judge_messages(item, response):
    prompt = JUDGE_PROMPT.format(question=item['question'], correct_answer=item['answer'], response=response)
    return [{'role': 'user', 'content': prompt}]


def valid_confidence(value):
    return type(value) is int and 0 <= value <= 100


def parse_judgment(text):
    """The judge's JSON as a dict with the schema's fields; raises ValueError when it does not conform."""
    import json
    if text is None:
        raise ValueError('judge returned no content')
    s = text.strip()
    if s.startswith('```'):
        s = s.strip('`')
        s = s[s.find('{'):]
    start, end = s.find('{'), s.rfind('}')
    if start < 0 or end < start:
        raise ValueError('judge content has no JSON object')
    obj = json.loads(s[start:end + 1])
    if obj.get('correct') not in ('yes', 'no'):
        raise ValueError(f"judge 'correct' is {obj.get('correct')!r}")
    for key in ('extracted_final_answer', 'reasoning'):
        if not isinstance(obj.get(key), str):
            raise ValueError(f'judge {key!r} missing')
    conf = obj.get('confidence')
    if not valid_confidence(conf):
        raise ValueError('judge confidence must be an integer in 0..100')
    return {'extracted_final_answer': obj['extracted_final_answer'], 'reasoning': obj['reasoning'],
            'correct': obj['correct'], 'confidence': conf}
