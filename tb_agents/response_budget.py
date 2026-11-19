"""Generated-output budgets, in the student's tokenizer, independent of prompt size.

One policy is shared by data synthesis and evaluation. Context guarding remains a
separate backstop. The body serializer is the pinned Snowball training template.
"""
import copy
import functools
import json
import os
import re
from pathlib import Path

TOKENIZER = '/data/aai/grug-b200/hf/ib-step18000-q175/tokenizer.json'
TEMPLATE = '/data/aai/tbenchtrain/sft_q35_code/8727101f2724d7caeba2a7753b509668f94c9f44/sft_snowball/snowball_chat_template.jinja'
GEN = '<|start_header_id|>assistant<|end_header_id|>\n<think>\n'
EOT = '<|eot_id|>'
CONTINUE = 'Thinking exceeded the maximum thinking budget, please continue.'
# Once the response budget's warning applies, a cut response is followed by CUT + the notice, never by CONTINUE:
# "please continue" contradicted "deliver your final answer now" (every warned W-v2 / Super-cap attempt got both).
# Pi itself inserts CONTINUE after a length stop; the proxy swaps it in the forwarded request (replace_continue),
# so the wire log, and SFT data built from it, carry the same text the model saw.
CUT = 'Your previous response was cut off at the response limit.'
POLICY = 'response-p80-plus-10pct-v3-finalization'
CAP_MARK = '[harbor-gen response cap reached; episode ended by the generation harness]'


@functools.lru_cache(1)
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(TOKENIZER)


SUPER_TOKENIZER = '/data/aai/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tokenizer.json'
SUPER_EOS = '<|im_end|>'


@functools.lru_cache(1)
def super_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(SUPER_TOKENIZER)


def super_text(text):
    """Generated text in the form the caps were measured on (experiments/super-response-caps, allowance-measurement):
    one native EOS per completed turn."""
    return text[:-len(EOT)] + SUPER_EOS if text.endswith(EOT) else text


def super_ids(text):
    return super_tokenizer().encode(super_text(text), add_special_tokens=False).ids


def super_clip(m, cap):
    """As clip, with the cap in Super tokens of the message body ('super' counting: one unit for every model, so the
    caps measured on Super mean the same amount of text for each)."""
    m = normalize(m)
    ids = super_ids(body(m))
    if len(ids) <= cap:
        return copy.deepcopy(m), encode(body(m)), False, len(ids)
    raw = super_tokenizer().decode(ids[:cap], skip_special_tokens=False).removesuffix(SUPER_EOS)
    if '</think>' in raw:
        reasoning, content = raw.split('</think>', 1)
        out = {'role': 'assistant', 'reasoning_content': reasoning.rstrip('\n'), 'content': content.lstrip('\n')}
    else:
        out = {'role': 'assistant', 'reasoning_content': raw, 'content': ''}
    return out, encode(body(out))[:-1], True, cap


@functools.lru_cache(1)
def template():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def fail(s):
        raise ValueError(s)
    env.globals['raise_exception'] = fail
    return env.from_string(Path(TEMPLATE).read_text())


def normalize(m):
    m = copy.deepcopy(m)
    if 'reasoning_content' not in m and isinstance(m.get('reasoning'), str):
        m['reasoning_content'] = m['reasoning']
    m.pop('reasoning', None)
    for tc in m.get('tool_calls') or []:
        f = tc.get('function', tc)
        if isinstance(f.get('arguments'), str):
            f['arguments'] = json.loads(f['arguments'])
    return m


def body(m):
    rendered = template().render(messages=[{'role': 'user', 'content': 'x'}, normalize(m)],
                                 bos_token='<|begin_of_text|>', add_generation_prompt=False)
    return rendered.split(GEN, 1)[1]


def encode(text):
    return tokenizer().encode(text, add_special_tokens=False).ids


def length(m):
    return len(encode(body(m)))


def clip(m, cap):
    """A partial model message + actual retained body ids. Never execute a partial tool call.

    Fabricated closing think/EOT tokens are serialization only and get no SFT loss.
    A clipped XML call remains plain content, so Pi cannot execute an incomplete call.
    """
    m = normalize(m)
    ids = encode(body(m))
    if len(ids) <= cap:
        return copy.deepcopy(m), ids, False
    ids = ids[:cap]
    raw = tokenizer().decode(ids, skip_special_tokens=False)
    if '</think>' in raw:
        reasoning, content = raw.split('</think>', 1)
        out = {'role': 'assistant', 'reasoning_content': reasoning.rstrip('\n'), 'content': content.lstrip('\n')}
    else:
        out = {'role': 'assistant', 'reasoning_content': raw, 'content': ''}
    return out, ids, True


def reserve(budget):
    return max(1, budget['finish_response_tokens']//10)


def hard_limit(budget):
    """Where generation stops: the total cap, or, in measurement mode ('measure_backstop_tokens'), warn + backstop.
    Measurement mode states the same notices (remaining = total - used, floored at 0) but never cuts at the stated
    allowance, so how many tokens a model actually takes to finish is observed uncensored."""
    b = budget.get('measure_backstop_tokens')
    return budget['warn_after_response_tokens'] + int(b) if b else budget['total_response_tokens']


NEAR_WINDOW_MARGIN = 512   # room for the CUT + notice user turns a near-window warning adds to the prompt


def item_budget(budget, max_output):
    """The budget for one single-turn item whose request may generate at most `max_output` tokens (its context room:
    window - prompt - margin, as the single-turn runners compute max_tokens). With 'near_window' set, the warning
    comes at whichever binds first: the suite cap, or max_output - allowance (the single-turn analogue of Pi's
    context finish notice); the total never exceeds max_output - NEAR_WINDOW_MARGIN. Otherwise unchanged."""
    if not budget or not budget.get('near_window') or not max_output:
        return budget
    allow = budget['finish_response_tokens']
    room = max(2, int(max_output) - NEAR_WINDOW_MARGIN)
    warn = max(1, min(budget['warn_after_response_tokens'], room - allow))
    total = min(budget['total_response_tokens'], max(warn + 1, room))
    return {**budget, 'warn_after_response_tokens': warn, 'finish_response_tokens': total - warn,
            'total_response_tokens': total, 'near_window_bound': warn < budget['warn_after_response_tokens']}


def output_cap(used, budget):
    if used < budget['warn_after_response_tokens']:
        return budget['warn_after_response_tokens']-used
    if budget.get('measure_backstop_tokens'):
        return max(1, hard_limit(budget)-used)
    left=budget['total_response_tokens']-used
    final=reserve(budget)
    return left-final if left>final else left


# Benchmark-specific sentence appended to the notice. Data synthesis and evaluation must both take it from here
# (via with_benchmark / the budget's 'notice_extra'), so a model is trained and evaluated on the same notice text.
# Omniscience: the generic "deliver your final answer now" overrode the task's own instruction that saying you do
# not know beats a wrong answer (W-v2: warned attempts 68% INCORRECT; the finishing data flipped 13% of Super's
# abstentions into answers). This restates the task's instruction; it adds no information.
NOTICE_EXTRA = {
    'omniscience': ('If you are not confident in an answer, say that you do not know: that is a valid final answer, '
                    'and it is better than a wrong one.'),
}


def with_benchmark(budget, benchmark):
    """The budget for `benchmark`, carrying its notice sentence (if any) as 'notice_extra'. Explicit values win."""
    if budget is None:
        return None
    out = dict(budget)
    if benchmark in NOTICE_EXTRA and 'notice_extra' not in out:
        out['notice_extra'] = NOTICE_EXTRA[benchmark]
    return out


def notice(budget, final=False, extra=None):
    return (f'[Harness notice] You have about {budget:,} response tokens left for the rest of this task, '
            'including reasoning and tool calls across all remaining turns. Tool results do not count against '
            'this allowance. When it runs out, the session ends and your work is graded as it stands. '
            'Finish your current line of reasoning, use tools only as needed, and deliver your final answer '
            'or deliverables now in the form and location the task asks for. Keep further reasoning brief. '
            f'For a text answer, aim for at most {max(1, budget//4):,} words and prioritize the essential requested information.'
            + (f' {extra}' if extra else '')
            + (' This is the reserved finalization allowance. Give your final answer or an honest concise status of '
               'the completed work now, then end the task. Do not start another repair or expand the work.' if final else ''))


_NOTICE_RE = re.compile(r'\[Harness notice\] You have about [\d,]+ response tokens left')


def _swap(text):
    """CONTINUE -> CUT in a text that carries a response-budget notice; anything else unchanged."""
    if isinstance(text, str) and _NOTICE_RE.search(text) and text.startswith(CONTINUE):
        return CUT + text[len(CONTINUE):]
    return text


def replace_continue(messages):
    """A new list where every user message that carries a response-budget notice has Pi's CONTINUE replaced by CUT
    (str content: leading CONTINUE; list content: a CONTINUE text part). Plain continuations (no notice, e.g. the
    16k per-response cap before the warning) and other notices (context limit) are untouched. Inputs not modified."""
    out = []
    for m in messages or []:
        if isinstance(m, dict) and m.get('role') == 'user':
            c = m.get('content')
            if isinstance(c, str):
                m = {**m, 'content': _swap(c)}
            elif isinstance(c, list):
                texts = ' '.join(p.get('text', '') for p in c if isinstance(p, dict))
                if _NOTICE_RE.search(texts):
                    c = [{**p, 'text': CUT} if isinstance(p, dict) and p.get('type') == 'text' and p.get('text') == CONTINUE
                         else p for p in c]
                    m = {**m, 'content': c}
        out.append(m)
    return out


def settings(env=None):
    env = os.environ if env is None else env
    if not env.get('PI_RESPONSE_BUDGET'):
        return None
    p = json.loads(env['PI_RESPONSE_BUDGET'])
    assert p['warn_after_response_tokens'] > 0 and p['finish_response_tokens'] > 0
    assert p['total_response_tokens'] == p['warn_after_response_tokens'] + p['finish_response_tokens']
    if p.get('token_counting', 'snowball') not in ('snowball', 'backend-native', 'super'):
        raise ValueError('Unsupported response-budget token_counting')
    return p
