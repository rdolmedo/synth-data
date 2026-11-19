"""Count complete, pre-parser generation and preserve truncated tool text safely."""
import json
import re
import uuid

from tb_agents import response_budget as B

ACCOUNTING_POLICY = 'pre-parser-generation-v1'


def canonical_text(text):
    # These are the EOS spellings of the pinned Super and Snowball backends.
    # A genuine generated EOS counts once in the common Snowball representation.
    for eos in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(eos):
            return text[:-len(eos)] + B.EOT
    return text


def invalid_arguments(message, tools):
    required = {t['function']['name']: t['function'].get('parameters', {}).get('required', [])
                for t in tools or []}
    for call in message.get('tool_calls') or []:
        function = call.get('function') or {}
        try:
            args = function.get('arguments')
            if isinstance(args, str):
                args = json.loads(args)
        except (ValueError, TypeError):
            return True
        if not isinstance(args, dict) or any(k not in args for k in required.get(function.get('name'), [])):
            return True
    return False


def unfinished_tool(text):
    return any(text.rfind(start) > text.rfind(end) for start, end in (
        ('<tool_call>', '</tool_call>'), ('<function=', '</function>'), ('<parameter=', '</parameter>')))


_CALL = re.compile(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', re.S)
# Datakit format (Marin's post-trained Snowball releases): <|start_think|>reasoning<|end_think|>content, then hermes-style
# JSON calls in <tool_call> blocks; the four tags are single special tokens, kept in the raw text (skip_special_tokens
# off). Only text carrying a Datakit think tag takes these paths, so Snowball/Super text parses exactly as before.
DK_START, DK_END = '<|start_think|>', '<|end_think|>'
_DK_CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)


def is_datakit(text):
    return DK_START in text or DK_END in text


def _dk_split(text):
    """(reasoning, rest) of Datakit text: the reasoning between <|start_think|> and the first <|end_think|>; a text
    without <|end_think|> is all reasoning when it opens with <|start_think|> (a cut inside the thinking)."""
    if DK_END in text:
        reasoning, rest = text.split(DK_END, 1)
        return reasoning.split(DK_START, 1)[-1], rest
    if text.lstrip().startswith(DK_START):
        return text.lstrip()[len(DK_START):], ''
    return '', text


def _parse_dk_calls(text, tools):
    reasoning, rest = _dk_split(text)
    first = rest.find('<tool_call>')
    if first < 0:
        return None
    calls, pos = [], first
    for m in _DK_CALL.finditer(rest, first):
        if rest[pos:m.start()].strip():
            return None
        try:
            obj = json.loads(m.group(1))
            name, args = obj['name'], obj.get('arguments', {})
            args = json.loads(args) if isinstance(args, str) else args
        except (ValueError, KeyError, TypeError):
            return None
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append({'id': 'chatcmpl-tool-' + uuid.uuid4().hex[:16], 'type': 'function',
                      'function': {'name': name, 'arguments': args}})
        pos = m.end()
    if not calls or rest[pos:].strip():
        return None
    message = {'role': 'assistant', 'reasoning_content': reasoning.strip('\n'), 'content': rest[:first].strip(),
               'tool_calls': calls}
    return None if invalid_arguments(message, tools) else message


_PARAM = re.compile(r'<parameter=([^>\n]+)>(.*?)</parameter>', re.S)


def _param_value(value, kind):
    # As vLLM's qwen3_coder parser: one newline after the open tag and before the close tag is layout.
    value = value[1:] if value.startswith('\n') else value
    value = value[:-1] if value.endswith('\n') else value
    if kind in (None, 'string'):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def parse_tool_text(text, tools=None):
    """An assistant message from raw generated text that ends in complete XML tool calls (the Snowball/Super
    format), or None when the calls are not clean (no call, unparsed text after the calls, a missing parameter)."""
    text = canonical_text(text).removesuffix(B.EOT)
    if is_datakit(text):
        return _parse_dk_calls(text, tools)
    reasoning, rest = text.split('</think>', 1) if '</think>' in text else ('', text)
    first = rest.find('<tool_call>')
    if first < 0:
        return None
    kinds = {t['function']['name']: {k: (v or {}).get('type') for k, v in
             ((t['function'].get('parameters') or {}).get('properties') or {}).items()} for t in tools or []}
    calls, pos = [], first
    for m in _CALL.finditer(rest, first):
        if rest[pos:m.start()].strip():
            return None
        name = m.group(1).strip()
        args = {pm.group(1).strip(): _param_value(pm.group(2), kinds.get(name, {}).get(pm.group(1).strip()))
                for pm in _PARAM.finditer(m.group(2))}
        calls.append({'id': 'chatcmpl-tool-' + uuid.uuid4().hex[:16], 'type': 'function',
                      'function': {'name': name, 'arguments': args}})
        pos = m.end()
    if not calls or rest[pos:].strip():
        return None
    message = {'role': 'assistant', 'reasoning_content': reasoning.rstrip('\n'), 'content': rest[:first].strip(),
               'tool_calls': calls}
    return None if invalid_arguments(message, tools) else message


def text_message(text, original):
    text = text.removesuffix(B.EOT)
    if is_datakit(text):
        reasoning, content = _dk_split(text)
        if reasoning or DK_END in text:
            return {'role': 'assistant', 'reasoning_content': reasoning.strip('\n'), 'content': content.lstrip('\n')}
    if '</think>' in text:
        reasoning, content = text.split('</think>', 1)
        return {'role': 'assistant', 'reasoning_content': reasoning.rstrip('\n'), 'content': content.lstrip('\n')}
    reasoning = original.get('reasoning_content') or original.get('reasoning') or ''
    if reasoning and not original.get('content') and not original.get('tool_calls'):
        return {'role': 'assistant', 'reasoning_content': text, 'content': ''}
    return {'role': 'assistant', 'reasoning_content': '', 'content': text}


def process(result, raw_text, cap, requested, budget, tools=None):
    """Return charged generation separately from the retained SFT message/IDs.

    The default budget unit remains Snowball tokens, but counts *raw generated*
    text before tool parsing. A backend-native budget must be explicitly selected
    and its thresholds computed in that same tokenizer. Cross-tokenizer conversion
    can exceed a Snowball boundary in one backend call; charge the excess instead
    of refunding it, and never generate again after the total is exhausted.
    """
    choice = result['choices'][0]
    native_ids = choice.get('token_ids')
    if not isinstance(native_ids, list) or any(type(i) is not int for i in native_ids):
        raise ValueError('Budgeted generation requires return_token_ids from the backend')
    if len(native_ids) > requested:
        raise ValueError('Backend generated more tokens than the requested native limit')
    reported = (result.get('usage') or {}).get('completion_tokens')
    if reported is not None and (type(reported) is not int or reported != len(native_ids)):
        raise ValueError('Backend completion usage disagrees with returned raw token IDs')
    raw_text = canonical_text(raw_text)
    raw_ids = B.encode(raw_text)
    counting = budget.get('token_counting', 'snowball')
    if counting not in ('snowball', 'backend-native', 'super'):
        raise ValueError('Unsupported response-budget token_counting')
    sup = B.super_ids(raw_text) if counting == 'super' else None
    generated = (len(native_ids) if counting == 'backend-native' else len(sup) if counting == 'super'
                 else len(raw_ids))
    # Detokenizers may omit a sole EOS. It still represents actual generation.
    if native_ids and not generated:
        generated = 1
    original = choice['message']
    native_limit = choice['finish_reason'] == 'length' or len(native_ids) >= requested
    truncated_tool = bool(original.get('tool_calls')) and (
        unfinished_tool(raw_text) or (native_limit and invalid_arguments(original, tools)))
    interrupted = generated > cap or choice['finish_reason'] == 'length' or truncated_tool
    if interrupted:
        # Never send an incomplete RPC to Pi. Keep the actual generated text as
        # non-executable context; Pi may continue only while total budget remains.
        text = (B.tokenizer().decode(raw_ids[:cap], skip_special_tokens=False) if counting == 'snowball'
                else B.super_tokenizer().decode(sup[:cap], skip_special_tokens=False).removesuffix(B.SUPER_EOS)
                if counting == 'super' else raw_text)
        message = text_message(text, original)
        if counting == 'snowball':
            message, ids, _ = B.clip(message, cap)
        else:
            ids = B.encode(B.body(message))
        choice['message'] = message
        choice['finish_reason'] = 'length'
    else:
        message = B.normalize(original)
        ids = B.encode(B.body(message))
        choice['message'] = message
    return {'message': message, 'ids': ids, 'clipped': interrupted,
            'charged_tokens': generated, 'native_tokens': len(native_ids),
            'raw_snowball_tokens': len(raw_ids), 'token_counting': counting,
            'native_limit_hit': native_limit, 'truncated_tool_call': truncated_tool,
            'accounting_policy': ACCOUNTING_POLICY}
