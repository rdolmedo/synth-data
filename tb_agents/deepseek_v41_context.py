"""Exact text rendering for the pinned V4.1 tokenizer's /tokenize endpoint.

vLLM's TokenizeChatRequest leaves tool_calls as Pydantic ValidatorIterator
objects, unlike ChatCompletionRequest. Its tokenizer then fails in deepcopy.
Render the same prompt from the JSON wire messages and use TokenizeCompletion
with add_special_tokens=False. The backend still performs all tokenization.
"""
import copy

from tb_agents.deepseek_v41_encoding import encode_messages


def tokenize_payload(payload):
    # The V4.1-Flash family shares this encoder and tokenizer (deepseek-v41-flash, deepseek-v41-flash-1017).
    if not str(payload.get('model', '')).startswith('deepseek-v41-flash'):
        raise ValueError('The pinned V4.1 renderer requires a deepseek-v41-flash model')
    messages = copy.deepcopy(payload['messages'])
    for message in messages:
        if 'reasoning' in message:
            message['reasoning_content'] = message['reasoning']
        content = message.get('content')
        if isinstance(content, list):
            parts = []
            for block in content:
                kind = block.get('type')
                if kind in ('text', 'input_text', 'output_text'):
                    parts.append(block.get('text', ''))
                else:
                    # Raw prompt tokenization cannot account for multimodal
                    # processor expansion. Pi advertises text-only input here;
                    # native images remain saved and quarantined by the exporter.
                    raise ValueError(f'Exact V4.1 text context cannot count {kind!r}')
            message['content'] = '\n\n'.join(parts)
    if payload.get('tools'):
        system = next((m for m in messages if m['role'] == 'system'), None)
        if system is None:
            system = {'role': 'system', 'content': ''}
            messages.insert(0, system)
        system['tools'] = copy.deepcopy(payload['tools'])
    kwargs = dict(payload.get('chat_template_kwargs') or {})
    # ChatCompletionRequest gives this top-level field precedence.
    effort = payload.get('reasoning_effort', kwargs.get('reasoning_effort', 'high'))
    thinking = bool(kwargs.get('thinking', kwargs.get('enable_thinking', True)))
    if effort == 'none':
        thinking, effort = False, None
    prompt = encode_messages(messages, thinking_mode='thinking' if thinking else 'chat',
        drop_thinking=kwargs.get('drop_thinking', True), reasoning_effort=effort)
    return {'model': payload['model'], 'prompt': prompt, 'add_special_tokens': False}
